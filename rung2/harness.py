"""
Rung-2 behavioral harness: ESCA's VLMPlanner (GroundingDINO + SGClip-with-our-heads
+ verified-SG prompt + gpt-4o-mini) drives a thin AI2-THOR-5.0 executor, with the
FROZEN chameleon patch injected onto the target object every frame it's visible.
Measures benign vs patched behavior under {base, esca, esca+language_only}.

Run (embench env):
  conda run -n embench python rung2/harness.py --task category_fridge
"""
import os, sys, json, argparse
NSAP, ESCA = "/home/usi/NSAP", "/home/usi/ESCA"
for p in (ESCA, NSAP, f"{NSAP}/src/LASER", f"{NSAP}/src/LASER/laser/models",
          f"{ESCA}/embodiedbench/evaluator", f"{NSAP}/rung2", f"{NSAP}/SceneGraphBench"):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if not os.environ.get("OPENAI_API_KEY"):
    import re
    t = open(f"{ESCA}/.openai_key").read()
    os.environ["OPENAI_API_KEY"] = re.search(r"sk-[A-Za-z0-9_-]+", t).group(0)
os.environ.setdefault("NSAP_GD_CKPT", f"{ESCA}/checkpoints/groundingdino_swint_ogc.pth")
os.chdir(NSAP)

import numpy as np, torch
from PIL import Image
from env import Rung2Env
from functions.attack.patch import paste_patch, compute_patch_size
from functions.attack.anchor import anchor_center, Anchor                 # noqa: F401
from functions.attack.train import AttackConfig                          # noqa: F401

from embodiedbench.planner.vlm_planner_esca import VLMPlanner
from embodiedbench.envs.eb_alfred.EBAlfEnv import get_global_action_space
from embodiedbench.evaluator.config.system_prompts import alfred_laser_system_prompt
from alfred_examples import laser_sg_examples_no_revert
import embodiedbench.evaluator.alfred.esca as ev_mod

ACTIONS = get_global_action_space()
TASKS = {t["id"]: t for t in json.load(open(f"{NSAP}/rung2/tasks.json"))["tasks"]}
# RELATIVE path under cwd (=NSAP): ESCA's planner splits the image path on '/' and
# os.path.join's it back to derive the bbox-viz path, which drops the leading slash
# on absolute paths. A relative path under an existing dir resolves correctly.
os.makedirs("output/rung2/obs", exist_ok=True)
TMP_IMG = "output/rung2/obs/obs.png"


def make_planner(scene_graph_text, language_only):
    return VLMPlanner(
        "gpt-4o-mini", "remote", ACTIONS, alfred_laser_system_prompt,
        examples=laser_sg_examples_no_revert, scene_graph_examples=ev_mod.target_examples,
        n_shot=2, obs_key="head_rgb", chat_history=False, language_only=language_only,
        use_feedback=True, multistep=0, tp=1, scene_graph_text=scene_graph_text,
        gd_only=False, top_k=1, aggr_thres=0.4, gd_box_threshold=0.25, gd_text_threshold=0.2,
        use_sam2=False, gd_checkpoint_path=os.environ["NSAP_GD_CKPT"])


def load_patch(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["w"], ck["anchor"], getattr(ck.get("config"), "coverage_ratio", 0.3)


def render_inject(env, target_id, patch, save_copy=None):
    """Save current THOR frame to TMP_IMG, injecting the patch on the target if visible."""
    ev = env.event
    frame = np.array(ev.frame)
    flipped_here = False
    if patch is not None:
        mask = ev.instance_masks.get(target_id)
        if mask is not None and int(mask.sum()) >= 600:
            ys, xs = np.where(mask)
            bbox = {"x1": int(xs.min()), "y1": int(ys.min()), "x2": int(xs.max()), "y2": int(ys.max())}
            w, anchor, cov = patch
            patch_hwc = (127.5 * (torch.tanh(w) + 1.0)).permute(1, 2, 0)
            ph, pw = compute_patch_size(bbox, cov)
            cx, cy = anchor_center(anchor, bbox, ph, pw)
            ft = torch.from_numpy(frame.astype(np.float32))
            frame = paste_patch(ft, patch_hwc, cx, cy, ph, pw).clamp(0, 255).numpy().astype(np.uint8)
            flipped_here = True
    Image.fromarray(frame).save(TMP_IMG)
    if save_copy:
        Image.fromarray(frame).save(save_copy)
    return TMP_IMG, flipped_here


def run_episode(task, planner, inject, tag, max_steps=20):
    import shutil
    # per-episode folder: every observation frame + the annotated view the MLLM saw
    ep_dir = f"output/rung2/frames/{task['id']}__{tag}"
    if os.path.isdir(ep_dir):
        shutil.rmtree(ep_dir)
    os.makedirs(ep_dir, exist_ok=True)

    env = Rung2Env(task["scene"], resolution=TASKS_RES)
    env.run_init(task.get("init"))
    target_id = (env._nearest(task["target_type"]) or {}).get("objectId")
    patch = load_patch(os.path.join(NSAP, task["patch"])) if inject else None

    planner.reset(); planner.set_actions(ACTIONS)
    instr = task["instruction"]

    fn = [0]   # frame counter
    def save_planner_view(idx):
        # ESCA writes the bbox/mask-annotated image it shows the MLLM here
        for v in ("output/rung2/obs/obs_mask_bbox.png", "output/rung2/obs/obs_bbox.png"):
            if os.path.exists(v):
                try: shutil.copy(v, f"{ep_dir}/frame{idx:02d}_planner.png")
                except Exception: pass
                return

    img_path, fl = render_inject(env, target_id, patch, save_copy=f"{ep_dir}/frame00_obs.png")
    fn[0] = 1
    steps = success = 0
    acted_on_target = False
    frames_with_patch = int(fl)
    trace = []
    ttype = task["target_type"].lower()
    while steps < max_steps and not success:
        decision_frame = fn[0] - 1
        try:
            action, _reason = planner.act(img_path, instr)
        except Exception as e:
            import traceback as _tb; _tb.print_exc()
            trace.append(("PLANNER_ERROR", False, repr(e)[:80])); break
        save_planner_view(decision_frame)   # what the MLLM actually saw for this decision
        if action == -2:
            trace.append(("empty_plan", False, "")); break
        if action == -1:
            steps += 1; trace.append(("invalid_json", False, "")); continue
        plan = action if isinstance(action, list) else [action]
        for a in plan:
            if steps >= max_steps:
                break
            lang = a if isinstance(a, str) else (ACTIONS[a] if isinstance(a, int) and 0 <= a < len(ACTIONS) else str(a))
            ok, fb = env.execute(lang); steps += 1
            if ttype in lang.lower() and ok:
                acted_on_target = True
            success = env.check_success(task["success"])
            planner.update_info({"action_id": a if isinstance(a, int) else 0,
                                 "env_feedback": fb, "last_action_success": float(ok),
                                 "task_success": float(success)})
            img_path, fl = render_inject(env, target_id, patch,
                                         save_copy=f"{ep_dir}/frame{fn[0]:02d}_obs.png")
            fn[0] += 1
            frames_with_patch += int(fl)
            trace.append((lang, bool(ok), fb))
            if success or not ok:
                break
    env.stop()
    result = {"success": bool(success), "steps": steps, "acted_on_target": acted_on_target,
              "frames_with_patch": frames_with_patch, "n_frames_saved": fn[0], "trace": trace}
    json.dump({"task": task["id"], "instruction": instr, "condition": tag, **result},
              open(f"{ep_dir}/episode.json", "w"), indent=2, default=str)
    return result


TASKS_RES = 600


CONDS = {  # cond -> (scene_graph_text, language_only, inject)
    "base_benign": (False, False, False), "base_patch": (False, False, True),
    "esca_benign": (True, False, False),  "esca_patch": (True, False, True),
    "lang_benign": (True, True, False),   "lang_patch": (True, True, True),
    # blind = no image AND no scene graph (instruction only) -> the perception floor
    "blind_benign": (False, True, False), "blind_patch": (False, True, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="category_fridge",
                    help="task id, comma list, or 'all'")
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--conditions", default="esca_benign,esca_patch,lang_benign,lang_patch")
    ap.add_argument("--tasks-file", default="rung2/tasks.json")
    args = ap.parse_args()
    global TASKS
    TASKS = {t["id"]: t for t in json.load(open(args.tasks_file))["tasks"]}
    task_ids = list(TASKS) if args.task == "all" else args.task.split(",")
    conds = args.conditions.split(",")
    os.makedirs(f"{NSAP}/output/rung2", exist_ok=True)

    planner_cache = {}   # (sg_text, language_only) -> VLMPlanner, reused across episodes
    def get_planner(sgt, lang):
        if (sgt, lang) not in planner_cache:
            print(f"   [loading planner sg_text={sgt} language_only={lang}]", flush=True)
            planner_cache[(sgt, lang)] = make_planner(sgt, lang)
        return planner_cache[(sgt, lang)]

    summary = {}
    for tid in task_ids:
        task = TASKS[tid]
        print(f"\n### TASK {tid}: \"{task['instruction']}\"  ({task['scene']}, {task['target_type']}, mode={task['mode']})", flush=True)
        res_path = f"{NSAP}/output/rung2/{tid}_result.json"
        results = json.load(open(res_path)) if os.path.exists(res_path) else {}
        summary[tid] = {}
        for cond in conds:
            sgt, lang, inj = CONDS[cond]
            planner = get_planner(sgt, lang)
            try:
                r = run_episode(task, planner, inj, cond, args.max_steps)
            except Exception as e:
                import traceback; traceback.print_exc()
                r = {"success": False, "steps": 0, "error": repr(e)[:120],
                     "acted_on_target": False, "frames_with_patch": 0, "n_frames_saved": 0, "trace": []}
            r["frames_dir"] = f"output/rung2/frames/{tid}__{cond}"
            results[cond] = r
            summary[tid][cond] = r["success"]
            print(f"  {cond:12} SUCCESS={r['success']}  steps={r['steps']}  acted_on_target={r.get('acted_on_target')}  patched_frames={r.get('frames_with_patch')}", flush=True)
            json.dump(results, open(res_path, "w"), indent=2, default=str)

    # ── summary table ──────────────────────────────────────────────────────────
    print("\n\n################  RUNG-2 BEHAVIORAL SWEEP  ################")
    hdr = ["task"] + conds
    print("  " + "".join(h.ljust(16) for h in hdr))
    for tid, cs in summary.items():
        row = [tid] + [("succ" if cs.get(c) else "FAIL") if c in cs else "-" for c in conds]
        print("  " + "".join(str(x).ljust(16) for x in row))
    json.dump(summary, open(f"{NSAP}/output/rung2/sweep_summary.json", "w"), indent=2)
    print("\nsaved output/rung2/sweep_summary.json + per-task results + per-step frames")


if __name__ == "__main__":
    main()
