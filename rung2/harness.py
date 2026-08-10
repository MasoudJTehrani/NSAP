"""
Rung-2 behavioral harness: ESCA's VLMPlanner (GroundingDINO + SGClip-with-our-heads
+ verified-SG prompt + gpt-4o-mini) drives a thin AI2-THOR-5.0 executor, with the
FROZEN chameleon patch injected onto the target object every frame it's visible.
Measures benign vs patched behavior under {base, esca, lang, blind} x {benign, patch}.

Run in the `embench` env. NOTE: `conda run -n embench` can fail if CONDA_EXE is unset
("__conda_exe: permission denied") -> call the env's python directly instead, e.g.
  <miniconda>/envs/embench/bin/python rung2/harness.py --task all \
      --tasks-file rung2/gap_tasks_v2.json --conditions base_benign,esca_benign,esca_patch
"""
import os, sys, json, argparse
NSAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESCA = os.environ.get("ESCA_ROOT", "/home/usi/ESCA")   # separate repo; override with $ESCA_ROOT
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
os.environ.setdefault("NSAP_VINE_WEIGHTS", f"{NSAP}/model/vine/model.safetensors")  # esca.py's default is machine-specific -> pin to this repo (trained VINE, not base CLIP)
os.chdir(NSAP)

import numpy as np, torch
from PIL import Image
from env import Rung2Env
from functions.attack.patch import paste_patch, compute_patch_size
from functions.attack.anchor import anchor_center, Anchor                 # noqa: F401
from functions.attack.train import AttackConfig                          # noqa: F401

from embodiedbench.planner.vlm_planner_esca import VLMPlanner
from embodiedbench.planner.alfred.esca import convert_gd_boxes_to_pixel
from embodiedbench.envs.eb_alfred.EBAlfEnv import get_global_action_space
from embodiedbench.evaluator.config.system_prompts import alfred_laser_system_prompt
from alfred_examples import laser_sg_examples_no_revert
import embodiedbench.evaluator.alfred.esca as ev_mod

ACTIONS = get_global_action_space() + [
    # stock EB-ALFRED lacks actions for these targets; Rung2Env.execute already
    # handles them, so expose them to the planner (else it falls back to action 0
    # = "find a Cart" and loops). Needed for toaster / coffee-machine tasks.
    "find a Toaster", "turn on the Toaster", "turn off the Toaster",
    "turn on the CoffeeMachine", "turn off the CoffeeMachine",
    "stop",   # ObjectNav-style arrival declaration: the agent ends the episode here
]
# TASKS is loaded in main() from --tasks-file (default rung2/tasks.json).
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


# THOR objectType -> the SGClip category the patch attacks (classify_vocab's lowercase
# names). Lets the gate ask "does VINE still call the target a <cat>?".
THOR2VINE = {"Pot": "pot", "Microwave": "microwave", "Toaster": "toaster",
             "CoffeeMachine": "coffee machine", "Fridge": "fridge", "Kettle": "kettle",
             "Apple": "apple", "Bowl": "bowl"}

# Manipulation verbs (navigation is excluded: moving toward the target is always allowed;
# only ACTING on it is perception-gated).
_INTERACTION_PREFIXES = ("pick up", "pickup", "open", "close", "turn on", "turn off", "slice")


def _is_target_interaction(lang, ttype):
    """True iff `lang` manipulates the target type (so it must pass the perception gate)."""
    a = (lang or "").strip().lower()
    return a.startswith(_INTERACTION_PREFIXES) and ttype in a


def _target_bbox(ev, target_id):
    """(bbox_dict, mask) of the target in the current frame, else (None, mask_or_None)."""
    mask = ev.instance_masks.get(target_id) if target_id else None
    if mask is None:
        return None, None
    ys, xs = np.where(mask)
    if not len(xs):
        return None, mask
    return {"x1": int(xs.min()), "y1": int(ys.min()),
            "x2": int(xs.max()), "y2": int(ys.max())}, mask


def _target_visible(env, target_id, min_area=600):
    """GT visibility (VINE-free, ObjectNav-style): is the target in the agent's view now?
    True if THOR's own `visible` flag is set, OR the target's instance mask is present with
    enough area (robust to the framing distance where THOR's ~1.5m `visible` reads false)."""
    if not target_id:
        return False
    o = env.obj_by_id(target_id)
    if o and o.get("visible"):
        return True
    m = env.event.instance_masks.get(target_id)
    return bool(m is not None and int(m.sum()) >= min_area)


def _paste_on_target(frame, bbox, patch):
    """Paste the patch on the target's bbox in full-frame coords. Returns (frame, flipped)."""
    if patch is None or bbox is None:
        return frame, False
    w, anchor, cov = patch
    ph, pw = compute_patch_size(bbox, cov)
    patch_hwc = (127.5 * (torch.tanh(w) + 1.0)).permute(1, 2, 0)
    cx, cy = anchor_center(anchor, bbox, ph, pw)
    ft = torch.from_numpy(frame.astype(np.float32))
    out = paste_patch(ft, patch_hwc, cx, cy, ph, pw).clamp(0, 255).numpy().astype(np.uint8)
    return out, True


def _whiten_outside_bbox(frame, bbox, alpha=1.0):
    """Blend everything OUTSIDE the target's bbox toward white (alpha=1 => hard white),
    leaving the bbox region (patch included) untouched. Removes background regions so GD +
    SGClip can't ground them AS the target -- kills the false-positive 'toaster' boxes on
    the counter/window that the patch can't reach -- while keeping the object at TRUE scale."""
    if bbox is None:
        return frame
    out = frame.astype(np.float32)
    keep = np.zeros(frame.shape[:2], dtype=bool)
    keep[bbox["y1"]:bbox["y2"] + 1, bbox["x1"]:bbox["x2"] + 1] = True
    white = np.full_like(out, 255.0)
    bg = ~keep
    out[bg] = out[bg] * (1.0 - alpha) + white[bg] * alpha
    return out.astype(np.uint8)


def vine_grounds_target(planner, env, target_id, patch, vine_cat, thr, save_to=None, whiten=False):
    """Man-in-the-middle perception gate. Render the CURRENT view with the patch on the
    target and ground it EXACTLY as the planner's act() does -- GroundingDINO detect over
    the fixed vocab, then SGClip classify each detected box -- and return (grounded,
    top_cat, conf). `grounded` = the target category is SGClip's argmax on some detected
    box with conf > thr (ESCA's keep-rule), i.e. the target IS in the scene graph the
    planner would build for this view. We deliberately use GD's boxes, NOT the GT
    instance-mask bbox: SGClip classifies GD's detection box for this object correctly
    (e.g. 'toaster'), while the tight GT-mask crop misreads it ('coffee machine'). So the
    gate sees what the planner sees; the patch denies the action only when it drops the
    target out of THAT grounding."""
    bbox, _mask = _target_bbox(env.event, target_id)
    frame, _ = _paste_on_target(np.array(env.event.frame), bbox, patch)
    if whiten:   # whiten the background so GD/SGClip can only ground the real target
        frame = _whiten_outside_bbox(frame, bbox)
    gd_vocab = [c for c in (list(getattr(planner, "classify_vocab", [])) +
                            list(getattr(planner, "common_objects", []))) if c != "other"]
    try:
        _labels, boxes = planner.predict_gd(frame, gd_vocab,
                                            box_threshold=planner.gd_box_threshold,
                                            text_threshold=planner.gd_text_threshold)
        if boxes is None or len(boxes) == 0:
            return False, None, 0.0            # nothing detected -> can't act on the target
        boxes_pixel = convert_gd_boxes_to_pixel(boxes, frame.shape[1], frame.shape[0])
        planner.predict_laser(image=frame, boxes=boxes_pixel, masks=None,
                              target_names=[vine_cat], threshold=thr)
    except Exception as e:
        print(f"  [gate] grounding failed ({repr(e)[:60]}) -> fail-open", flush=True)
        return True, None, 0.0                 # never block the pipeline on a grounding error
    grounded, top_cat, conf = False, None, 0.0
    for _b, cat, c in getattr(planner, "last_boxes_full", []):
        if cat == vine_cat and c > thr:
            grounded = True
        if c >= conf:
            top_cat, conf = cat, c             # global argmax, just for the log line
    if save_to:
        try:
            from PIL import ImageDraw
            im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
            for _b, cat, c in getattr(planner, "last_boxes_full", []):
                x1, y1, x2, y2 = (int(v) for v in _b)
                col = (60, 220, 60) if cat == vine_cat else (255, 60, 60)
                dr.rectangle([x1, y1, x2, y2], outline=col, width=2)
                dr.text((x1 + 2, max(0, y1 - 11)), f"{cat} {c:.2f}", fill=col)
            im.save(save_to)
        except Exception:
            pass
    return grounded, top_cat, conf


def render_inject(env, target_id, patch, save_copy=None, whiten=False):
    """Save the current THOR frame to TMP_IMG for the planner, with the patch injected on
    the target in full-frame coords. If whiten, the PLANNER INPUT has its background whitened
    (so VINE's SG only grounds the real target); the obs save_copy stays the full frame."""
    ev = env.event
    # patch EVERY frame the target is present, regardless of size (paste in full-frame
    # coords; the size floor / degenerate-patch guard were removed for testing).
    bbox, mask = _target_bbox(ev, target_id)
    frame, flipped_here = _paste_on_target(np.array(ev.frame), bbox, patch)
    fed = _whiten_outside_bbox(frame, bbox) if whiten else frame
    Image.fromarray(fed).save(TMP_IMG)            # planner / VINE input (bg whitened if enabled)
    if save_copy:
        Image.fromarray(frame).save(save_copy)    # obs view = full patched frame (never whitened)
    return TMP_IMG, flipped_here


def _nav_grounded(planner, success_spec, vine_cat, thr):
    """Navigation 'seen' check, matched to the attack TYPE the task encodes:
      - spec has 'attribute' -> VINE grounds (attribute, target)   [_attr tasks]
      - spec has 'relation'  -> VINE grounds (target, relation, *)  [_rel tasks]
      - else                  -> VINE grounds the target CATEGORY    [_cat tasks]
    Category uses SGClip's per-box argmax (last_boxes_full); attribute/relation read the
    'Perception-verified scene graph' text ('(attr, obj)' / '(subj, pred, obj)' format)."""
    if success_spec.get("attribute"):
        sg = (getattr(planner, "last_sg_text", "") or "").lower()
        return f"({success_spec['attribute'].lower()}, {vine_cat})" in sg
    if success_spec.get("relation"):
        sg = (getattr(planner, "last_sg_text", "") or "").lower()
        return f"({vine_cat}, {success_spec['relation'].lower()}, " in sg
    return any(cat == vine_cat and c > thr
               for _b, cat, c in getattr(planner, "last_boxes_full", []))


def run_episode(task, planner, inject, tag, max_steps=14, isolate=True, closed_loop=True, whiten=False):
    import shutil
    # per-episode folder: every observation frame + the annotated view the MLLM saw
    ep_dir = f"output/rung2/frames/{task['id']}__{tag}"
    if os.path.isdir(ep_dir):
        shutil.rmtree(ep_dir)
    os.makedirs(ep_dir, exist_ok=True)

    env = Rung2Env(task["scene"], resolution=TASKS_RES)
    env.run_init(task.get("init"))
    target_id = (env._nearest(task["target_type"]) or {}).get("objectId")
    # MIM isolation test: strip all non-target objects so no confusable item can be
    # (mis)grounded as the target category alongside the real, patched one.
    if isolate and target_id:
        n_dis = env.isolate_target(target_id)
        print(f"  [isolate] removed {n_dis} movable objects (kept {task['target_type']} + structure)", flush=True)
    patch = load_patch(os.path.join(NSAP, task["patch"])) if inject else None
    vine_cat = THOR2VINE.get(task["target_type"], task["target_type"].lower())
    gate_thr = getattr(planner, "aggr_thres", 0.4)

    planner.reset(); planner.set_actions(ACTIONS)
    instr = task["instruction"]

    fn = [0]   # frame counter
    def save_planner_view(idx):
        # base conditions (scene_graph_text=False) don't USE VINE's grounding in the
        # decision, so a "planner view" there is misleading -> skip it. (VINE still RUNS
        # in base -- the grounding loop is unconditional in ESCA -- but its output is
        # ignored by the MLLM, so there's nothing meaningful to visualize.)
        if not getattr(planner, "scene_graph_text", True):
            return
        # Draw SGClip's REAL per-box (category, conf) on the frame the planner ACTUALLY
        # grounded (TMP_IMG = the full observation), so boxes align and this view shows
        # exactly what VINE saw.
        src = TMP_IMG
        if not os.path.exists(src):
            return
        try:
            from PIL import ImageDraw
            im = Image.open(src).convert("RGB"); dr = ImageDraw.Draw(im)
            for bbox, cat, conf in getattr(planner, "last_boxes_full", []):
                x1, y1, x2, y2 = (int(v) for v in bbox)
                dr.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=2)
                dr.text((x1 + 2, max(0, y1 - 10)), f"{cat} {conf:.2f}", fill=(255, 60, 60))
            im.save(f"{ep_dir}/frame{idx:02d}_planner.png")
        except Exception:
            pass

    img_path, fl = render_inject(env, target_id, patch, save_copy=f"{ep_dir}/frame00_obs.png", whiten=whiten)
    fn[0] = 1
    steps = success = 0
    acted_on_target = False
    stopped = False              # did the robot issue 'stop' (declare arrival)?
    vine_grounded_any = False    # did VINE ever confirm the target? (LOGGED signal, not the decision)
    frames_with_patch = int(fl)
    trace = []
    sg_log = []   # per decision: scene-graph text the planner saw + its own object list + action
    ttype = task["target_type"].lower()
    while steps < max_steps and not stopped:
        decision_frame = fn[0] - 1
        try:
            action, _reason = planner.act(img_path, instr)
        except Exception as e:
            import traceback as _tb; _tb.print_exc()
            trace.append(("PLANNER_ERROR", False, repr(e)[:80])); break
        save_planner_view(decision_frame)   # what the MLLM actually saw for this decision
        _sgi = getattr(planner, "last_sg_info", {}) or {}
        try:                       # _reason is the full MLLM plan JSON (esca.py act() line 2635)
            _plan = json.loads(_reason)
        except Exception:
            _plan = _reason
        # LOG-ONLY: does VINE confirm the target this step (category/attr/rel per the task)?
        # The success DECISION is the robot's own 'stop' below, NOT this -- we record it so we
        # can compare "VINE grounded it" against "the robot acted on it".
        vine_conf = _nav_grounded(planner, task["success"], vine_cat, gate_thr)
        vine_grounded_any = vine_grounded_any or vine_conf
        sg_log.append({"step": decision_frame,
                       "scene_graph": getattr(planner, "last_sg_text", "") or "",
                       "mllm_objects": _sgi.get("current_objects"),
                       "vine_confirmed": vine_conf,
                       # full planner output: reasoning_and_reflection, language_plan, executable_plan, scene_graph
                       "plan": _plan,
                       "action": action if isinstance(action, (str, list)) else str(action)})
        if action == -2:
            trace.append(("empty_plan", False, "")); break
        if action == -1:
            steps += 1; trace.append(("invalid_json", False, "")); continue
        plan = action if isinstance(action, list) else [action]
        for a in plan:
            if steps >= max_steps:
                break
            lang = a if isinstance(a, str) else (ACTIONS[a] if isinstance(a, int) and 0 <= a < len(ACTIONS) else str(a))
            # STOP = the robot declares it has arrived (ObjectNav-style). We judge the AGENT'S
            # action: success iff the target is actually GT-visible now, regardless of whether
            # VINE, the MLLM's own vision, or the instruction drove the stop.
            if lang.strip().lower().startswith("stop"):
                vis = _target_visible(env, target_id)
                success, stopped = (1 if vis else 0), True
                trace.append((lang, bool(vis),
                              "stop: target VISIBLE -> success" if vis else "stop: NOT visible -> fail"))
                break
            ok, fb = env.execute(lang); steps += 1
            if ttype in lang.lower() and ok:
                acted_on_target = True
            planner.update_info({"action_id": a if isinstance(a, int) else 0,
                                 "env_feedback": fb, "last_action_success": float(ok),
                                 "task_success": 0.0})
            img_path, fl = render_inject(env, target_id, patch,
                                         save_copy=f"{ep_dir}/frame{fn[0]:02d}_obs.png", whiten=whiten)
            fn[0] += 1
            frames_with_patch += int(fl)
            trace.append((lang, bool(ok), fb))
            if not ok:
                break
    env.stop()
    # metric B cross-check (human-readable): did the target ever appear in VINE's SG text /
    # the MLLM's plan? "mentioned" != "found" -- use alongside metric A, not instead of it.
    def _plan_acts(e):
        p = e.get("plan")
        ep = p.get("executable_plan") if isinstance(p, dict) else None
        return " ".join(a.get("action_name", "") if isinstance(a, dict) else str(a)
                        for a in (ep or []))
    target_in_sg   = any(vine_cat in (e.get("scene_graph", "") or "").lower() for e in sg_log)
    target_in_plan = any(ttype in _plan_acts(e).lower() for e in sg_log)
    result = {"success": bool(success), "stopped": stopped, "steps": steps,
              "acted_on_target": acted_on_target,
              # VINE-vs-behavior: did VINE ever confirm the target, and did the plan/SG mention
              # it -- logged for comparison; the success DECISION is the robot's stop above.
              "vine_grounded_any_step": vine_grounded_any,
              "target_in_sg_any_step": target_in_sg, "target_in_plan_any_step": target_in_plan,
              "frames_with_patch": frames_with_patch, "n_frames_saved": fn[0], "trace": trace,
              "sg_log": sg_log}
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
    ap.add_argument("--task", default="all",
                    help="task id, comma list, or 'all' (default: every task in --tasks-file)")
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--conditions", default="esca_benign,esca_patch,lang_benign,lang_patch")
    ap.add_argument("--tasks-file", default="rung2/tasks.json")
    ap.add_argument("--no-isolate", action="store_false", dest="isolate",
                    help="Disable MIM isolation (default ON). Keeps ALL scene objects "
                         "instead of stripping everything but the target (+support).")
    ap.add_argument("--full-plan", action="store_false", dest="closed_loop",
                    help="Disable the perception gate (open-loop): let target interactions "
                         "resolve on the GT object regardless of VINE. Default is gated -- a "
                         "pick-up/open/toggle of the target only runs if VINE still grounds it "
                         "in the current (patched) view, so the patch can deny the action.")
    ap.add_argument("--language-only", action="store_true", dest="language_only",
                    help="Force the planner BLIND (no image) for every condition, so it must "
                         "rely on VINE's grounded scene graph -- the setting where the "
                         "perception attack can actually change the plan.")
    ap.add_argument("--whiten-bg", action="store_true", dest="whiten",
                    help="Whiten everything OUTSIDE the target's GT bbox in VINE's input "
                         "(planner grounding AND the gate), so GD/SGClip can't ground "
                         "background regions AS the target. Kills the immovable false-positive "
                         "'toaster'/'pot' boxes the patch can't reach. (Helps toaster; note it "
                         "can erase context-dependent targets like the pot.)")
    args = ap.parse_args()
    global TASKS
    _data = json.load(open(args.tasks_file))
    TASKS = {t["id"]: t for t in _data["tasks"]}
    file_max_steps = _data.get("max_steps", args.max_steps)   # file-level default; a task can override
    task_ids = list(TASKS) if args.task == "all" else args.task.split(",")
    missing = [t for t in task_ids if t not in TASKS]
    if missing:
        raise SystemExit(f"[harness] unknown task id(s) {missing}. "
                         f"Available in {args.tasks_file}: {list(TASKS)}")
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
            if args.language_only:   # force the blind planner regardless of the cond's default
                lang = True
            planner = get_planner(sgt, lang)
            try:
                r = run_episode(task, planner, inj, cond, task.get("max_steps", file_max_steps),
                                isolate=args.isolate, closed_loop=args.closed_loop,
                                whiten=task.get("whiten", args.whiten))
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
