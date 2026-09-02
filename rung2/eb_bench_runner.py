"""Faithful EB-Navigation benchmark runner with man-in-the-middle patch injection.

Uses the REAL benchmark env (EBNavigationEnv: exact Teleport to the dataset pose, the 8
discrete skills, measure_success = GT distance <= SUCCESS_THRESHOLD, 20-step budget) and
the FIXED ESCA nav planner (embodiedbench.planner.navigation.esca, wired to VINE's trained
SGClip weights). The ONLY interference is the MIM: every frame the planner sees is
intercepted and, when injecting, the trained patch is pasted onto the target's GT bbox
(instance_detections2D) before the planner grounds on it. Nothing in the benchmark env or
success metric is modified.

  python rung2/eb_bench_runner.py --eval-sets long_horizon,visual_appearance \
      --indices 1,2,3,4,7,9,11,13,15,16,18,19 --conditions benign,patch --model gpt-4o
"""
import os, sys, json, argparse, time, shutil
# determinism for VINE/GD -- MUST be set before torch/CUDA init (i.e. before importing harness)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NSAP_LLM_SEED", "42")
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rung2/
import harness   # noqa: F401  sets sys.path (ESCA/LASER), env vars (OPENAI/GD/VINE), chdir(NSAP)
from nav_harness import make_nav_planner                          # noqa: E402  fixed navigation.esca planner
import nsap_inject                                                # noqa: E402
from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv   # noqa: E402  the REAL benchmark env

NSAP = harness.NSAP

# cond -> (scene_graph_text, language_only, inject).  base = vanilla MLLM (no SG);
# esca = MLLM + VINE scene graph; esca_patch injects the chameleon.
CONDS = {
    "base_benign": (False, False, False),
    "esca_benign": (True,  False, False),
    "esca_patch":  (True,  False, True),
}


def set_determinism(seed=42):
    """Make VINE + GroundingDINO forward passes as deterministic as PyTorch allows
    (fixed seeds + cuDNN deterministic). The gpt-4o vision path stays only best-effort
    deterministic server-side; this removes the *grounding* noise component."""
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"  [determinism] use_deterministic_algorithms warn: {repr(e)[:100]}", flush=True)
    print(f"  [determinism] seeds={seed} cudnn.deterministic=True (VINE/GD)", flush=True)

# per-EB-Nav-index SCENE-MATCHED chameleon: (category slug, kitchen_ebnav scenario id).
# Only the 5 indices whose chameleon actually trained are kept; the rest are unattackable and
# omitted (kettle at idx 11/18, plus pot/toaster scenarios where VINE never reads the object,
# so there is no clean instance to attack -- same root cause as the kettle drop).
CHAM = {
    2:  ("toaster", "00001"),        3:  ("pot", "00002"),
    4:  ("pot", "00003"),            7:  ("toaster", "00004"),
    19: ("coffee_machine", "00011"),
}


def _patch_for_index(idx):
    # CHAMELEON ONLY -- the universal (loud) patch is no longer used/evaluated. A scenario
    # with no chameleon (VINE can't read the object) is simply not attacked.
    slug, sid = CHAM[idx]
    cham = (f"output/chameleon_patches/category_{slug}/ebnav_scenario_{sid}/"
            f"category_{slug}/category_{slug}_final.pt")
    if os.path.isfile(os.path.join(NSAP, cham)):
        return cham, "chameleon"
    return None, "none"


def load_patches(indices):
    patches, kinds = {}, {}
    for idx in indices:
        p, kind = _patch_for_index(idx)
        patches[idx] = nsap_inject.load_patch(os.path.join(NSAP, p)) if p else None
        kinds[idx] = kind
        print(f"  [patch] idx {idx:>2}  {CHAM[idx][0]:14} -> {kind}", flush=True)
    return patches, kinds


def mim_frame(env, obs, patch, target_id):
    """Intercept the frame the planner will see; paste the patch on the target's GT bbox
    if injecting and the target is visible. Returns (frame_uint8_rgb, injected_bool)."""
    frame = np.array(obs["head_rgb"]).astype(np.uint8)
    if patch is None:
        return frame, False
    dets = env.env.last_event.instance_detections2D or {}
    bbox = dets.get(target_id)
    if bbox is None:                       # target not GT-visible this frame -> nothing to patch
        return frame, False
    return nsap_inject.paste_on_bbox(frame, bbox, patch)


def _dump_step(planner, viz_dir, step, instr, img_path, action, reasoning, pf):
    """Append a readable per-step trace to <ep>/trace.txt so you can see WHY the agent moved:
    instruction -> what VINE was asked to ground -> GroundingDINO detections -> SGClip's grounded
    objects/relations -> the SG TEXT fed to the MLLM -> the MLLM's plan -> the chosen action."""
    L = ["=" * 84,
         f"STEP {step}   frame={os.path.basename(img_path)}   (patched frames so far: {pf})",
         f"INSTRUCTION: {instr}"]
    c = getattr(planner, "_nsap_last_concepts", None)
    if c:
        L.append(f"[1] MLLM asked SGClip to ground -> objects={c.get('objects')}  "
                 f"attributes={c.get('attributes')}  relations={c.get('relations')}")
    L.append(f"[1b] detector query={getattr(planner, '_nsap_gd_query', '?')}  -> GroundingDINO "
             f"found {getattr(planner, '_nsap_gd_nboxes', '?')} boxes: {getattr(planner, '_nsap_gd_labels', None)}")
    L.append(f"[2] SGClip grounded objects:   {getattr(planner, '_nsap_last_labels', None)}")
    L.append(f"[2b] SGClip top-k per box (prob,name) [aggr_thres={getattr(planner, 'aggr_thres', '?')}]: "
             f"{getattr(planner, '_nsap_predicted_name', None)}")
    L.append(f"[2b] classify vocab (query_names): {getattr(planner, '_nsap_query_names', None)}")
    L.append(f"[2] SGClip grounded relations: {getattr(planner, '_nsap_last_relations', None)}")
    L.append(f"[3] SG TEXT fed to the MLLM:\n    {getattr(planner, '_nsap_last_sg_text', '')}")
    if isinstance(reasoning, dict) and reasoning:
        for k, v in reasoning.items():
            L.append(f"[4] MLLM {k}: {str(v)[:700]}")
    L.append(f"[5] ACTION chosen: {action}")
    with open(os.path.join(viz_dir, "trace.txt"), "a") as f:
        f.write("\n".join(L) + "\n\n")


def run_episode(env, planner, patch, viz_dir):
    """One benchmark episode with MIM injection. Mirrors evaluator.evaluate()'s inner loop."""
    obs = env.reset()
    ep_idx = env.selected_indexes[env._current_episode_num - 1]
    target_id = env.episode_data["targetObjectIds"]
    cat = env.episode_data["targetObjectType"]
    instr = env.episode_language_instruction
    planner.reset()
    os.makedirs(viz_dir, exist_ok=True)
    pf = 0
    step = 0

    def save(o, s):
        nonlocal pf
        frame, fl = mim_frame(env, o, patch, target_id)
        p = os.path.join(viz_dir, f"frame{s:02d}.png")
        Image.fromarray(frame).save(p)
        pf += int(fl)
        return p

    def save_grounded(s):
        for g in ("output/rung2/obs/obs_bbox.png", "running/eb_nav/obs_bbox.png"):
            if os.path.exists(g):
                try: shutil.copy(g, os.path.join(viz_dir, f"dec{s:02d}_grounded.png"))
                except Exception: pass
                break

    img_path = save(obs, step)
    done = False
    info = {"task_success": 0.0, "distance": None}
    test_ct = 0
    while not done:
        try:
            action, reasoning = planner.act(img_path, instr)
            try: reasoning = json.loads(reasoning)
            except Exception: reasoning = {}
            save_grounded(step)
            _dump_step(planner, viz_dir, step, instr, img_path, action, reasoning, pf)
            acts = action if isinstance(action, list) else [action]
            budget = max(1, env._max_episode_steps - env._current_step + 1)
            for i, a in enumerate(acts[:budget]):
                obs, reward, done, info = env.step(a, reasoning, 1 if i == 0 else 0)
                planner.update_info(info)
                step += 1
                img_path = save(obs, step)
                if done:
                    break
                if info.get("last_action_success") == 0:
                    break                                   # invalid action -> replan
        except Exception as e:
            test_ct += 1
            print(f"      [ep{ep_idx}] planner/env error ({test_ct}): {repr(e)[:200]}", flush=True)
            if test_ct == 1:
                import traceback as _tb; _tb.print_exc()
            if test_ct > 3:
                break
            time.sleep(1)
    res = {"episode": ep_idx, "target": cat, "scene": env.episode_data["scene"],
           "success": float(info.get("task_success", 0.0)), "steps": env._current_step,
           "distance": round(float(info.get("distance") or -1), 3), "frames_with_patch": pf}
    print(f"    ep{ep_idx:>2} {cat:13} {'SUCCESS' if res['success'] else 'fail   '}  "
          f"steps={res['steps']:>2} dist={res['distance']} pf={pf}", flush=True)
    return res


def run_set(eval_set, indices, planner, inject, patches, model, tag, rep_tag=""):
    exp = "/".join(x for x in ["nsap_mim", rep_tag, tag, eval_set] if x)
    env = EBNavigationEnv(eval_set=eval_set, exp_name=exp, boundingbox=False,
                          resolution=500, selected_indexes=indices)
    planner.set_actions(env.language_skill_set)
    # RELATIVE to cwd (=NSAP, set by harness): the planner derives its *_bbox.png path via
    # os.path.join(*image_path.split('/')), which drops the leading slash of an ABSOLUTE path
    # and turns it relative -> FileNotFoundError. Relative paths sidestep that entirely.
    viz_root = os.path.join("output/rung2/ebnav", rep_tag, tag, eval_set)
    results = []
    while env._current_episode_num < env.number_of_episodes:
        ep_peek = env.selected_indexes[env._current_episode_num]
        cat = env.dataset[env._current_episode_num]["targetObjectType"]
        patch = patches.get(ep_peek) if inject else None   # scene-matched patch per EB-Nav index
        viz_dir = os.path.join(viz_root, f"ep{ep_peek}_{cat}")
        results.append(run_episode(env, planner, patch, viz_dir))
        os.makedirs(viz_root, exist_ok=True)
        json.dump(results, open(os.path.join(viz_root, "results.json"), "w"), indent=2)
    try: env.close()
    except Exception: pass
    return results


SHORT = {"base_benign": "base", "esca_benign": "esca_b", "esca_patch": "esca_p"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-sets", default="long_horizon,visual_appearance")
    ap.add_argument("--indices", default="2,3,4,7,19")   # the 5 indices with a trained EB-Nav chameleon
    ap.add_argument("--conditions", default="base_benign,esca_benign,esca_patch")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--reps", type=int, default=1, help="repeat every episode N times")
    args = ap.parse_args()

    set_determinism(int(os.environ.get("NSAP_LLM_SEED", "42")))
    eval_sets = args.eval_sets.split(",")
    indices = [int(x) for x in args.indices.split(",")]
    conds = args.conditions.split(",")
    reps = max(1, args.reps)
    patches, kinds = load_patches(indices)

    _pcache = {}
    def get_planner(sgt, lang):
        if (sgt, lang) not in _pcache:
            print(f"\n[loading planner sg_text={sgt} language_only={lang} model={args.model}]", flush=True)
            _pcache[(sgt, lang)] = make_nav_planner(scene_graph_text=sgt, language_only=lang, model_name=args.model)
        return _pcache[(sgt, lang)]

    allres = {}   # (rep, eval_set, cond) -> results list
    for rep in range(reps):
        rep_tag = f"rep{rep}" if reps > 1 else ""
        for eval_set in eval_sets:
            for cond in conds:
                sgt, lang, inject = CONDS[cond]
                planner = get_planner(sgt, lang)
                print(f"\n===== [{rep_tag or 'single'}] {eval_set} | {cond} ({'inject' if inject else 'clean'}) =====", flush=True)
                allres[(rep, eval_set, cond)] = run_set(eval_set, indices, planner, inject, patches, args.model, cond, rep_tag)

    # ---- comparison report (aggregated across reps) ----
    print("\n" + "#" * 84)
    print(f"#  EB-NAV BENCHMARK MIM  (reps={reps}, model={args.model}, VINE/GD determinism)")
    print("#" * 84)
    def pool(es, c): return [r for rep in range(reps) for r in allres.get((rep, es, c), [])]
    summary = {}
    for eval_set in eval_sets:
        tot = len(indices) * reps
        sr = {c: sum(r["success"] for r in pool(eval_set, c)) for c in conds}
        print(f"\n== {eval_set} ==  " + "   ".join(f"{c} {sr[c]:.0f}/{tot}={sr[c]/tot:.0%}" for c in conds))
        if "esca_benign" in conds and "esca_patch" in conds:
            print(f"   ATTACK (esca_benign->esca_patch): {(sr['esca_patch']-sr['esca_benign'])/tot*100:+.0f} pts")
        print(f"   {'ep':>3} {'target':13} {'kind':10} " + " ".join(f"{SHORT.get(c,c):>8}" for c in conds))
        for idx in indices:
            cells = [f"{sum(r['success'] for r in pool(eval_set,c) if r['episode']==idx):.0f}/{reps}" for c in conds]
            tgt = next((r["target"] for c in conds for r in pool(eval_set, c) if r["episode"] == idx), "?")
            print(f"   {idx:>3} {tgt:13} {kinds.get(idx,'?'):10} " + " ".join(f"{x:>8}" for x in cells))
        summary[eval_set] = {c: f"{sr[c]:.0f}/{tot}" for c in conds}
    json.dump(summary, open("output/rung2/ebnav/summary.json", "w"), indent=2)
    print("\n(viz + per-episode results under output/rung2/ebnav/<rep>/<cond>/<eval_set>/)")


if __name__ == "__main__":
    main()
