#!/usr/bin/env python
"""Navigation MIM harness -- EmbodiedBench EB-Navigation style.

Real step-by-step movement (Move/Rotate/Look), success = GT distance(agent,target) <= 1 m
(EBNavEnv.SUCCESS_THRESHOLD), auto-terminated -- no stop action. The adversarial patch is
injected on the target whenever it is GT-visible as the agent walks toward it, so a
corrupted scene graph makes a VINE-reliant agent (esca/lang) mis-navigate and never reach.
Uses ESCA's own EBNavigationPlanner (movement actions + VINE scene graph).

Run (embench python):
  <miniconda>/envs/embench/bin/python rung2/nav_harness.py --tasks-file rung2/gap_tasks_v2.json \
      --task pot_cat --conditions base_benign,esca_benign,lang_benign,lang_patch
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness   # noqa: F401  -- sets sys.path (ESCA/LASER), OPENAI/GD/VINE env vars, os.chdir(NSAP)
from harness import render_inject, load_patch, THOR2VINE, NSAP   # noqa: E402
from env import Rung2Env                                          # noqa: E402
import numpy as np                                                # noqa: E402,F401

from embodiedbench.planner.navigation.esca import EBNavigationPlanner                 # noqa: E402
from embodiedbench.evaluator.config.system_prompts import eb_laser_navigation_system_prompt  # noqa: E402
from embodiedbench.evaluator.config.eb_navigation_example import (                    # noqa: E402
    examples as NAV_EX, laser_sg_examples_no_revert as NAV_SG_EX, target_examples as NAV_TGT_EX)

NAV_SKILLS = Rung2Env.NAV_SKILLS
SUCCESS_THRESHOLD = 1.0   # meters (EB-Navigation)

CONDS = {  # cond -> (scene_graph_text, language_only, inject)
    "base_benign": (False, False, False), "base_patch": (False, False, True),
    "esca_benign": (True, False, False),  "esca_patch": (True, False, True),
    "lang_benign": (True, True, False),   "lang_patch": (True, True, True),
}


def make_nav_planner(scene_graph_text, language_only, model_name="gpt-4o"):
    # GroundingDINO config ships inside the groundingdino package (alfred/esca.py resolves it
    # the same way); the nav planner takes it as an explicit arg, so point it there.
    import groundingdino
    gd_cfg = os.path.join(os.path.dirname(groundingdino.__file__),
                          "config", "GroundingDINO_SwinT_OGC.py")
    return EBNavigationPlanner(
        model_name=model_name, model_type="remote",
        actions=NAV_SKILLS, system_prompt=eb_laser_navigation_system_prompt,
        examples=(NAV_SG_EX if scene_graph_text else NAV_EX), scene_graph_examples=NAV_TGT_EX,
        n_shot=2, obs_key="head_rgb", chat_history=False, language_only=language_only,
        scene_graph_text=scene_graph_text, gd_only=False, tp=1, top_k=1, aggr_thres=0.4,
        gd_config_path=gd_cfg, gd_checkpoint_path=os.environ["NSAP_GD_CKPT"])


def run_episode(task, planner, inject, tag, max_steps=20, seed=0, success_dist=1.0):
    import shutil
    ep_dir = f"output/rung2/nav/{task['id']}__{tag}"
    if os.path.isdir(ep_dir):
        shutil.rmtree(ep_dir)
    os.makedirs(ep_dir, exist_ok=True)

    env = Rung2Env(task["scene"], resolution=600)
    env.run_init(task.get("init"))
    target_id = (env._nearest(task["target_type"]) or {}).get("objectId")
    if not target_id:
        env.stop()
        return {"success": False, "error": f"no {task['target_type']} in scene"}
    patch = load_patch(os.path.join(NSAP, task["patch"])) if inject else None
    start_dist = env.nav_start(target_id, min_dist=2.5, seed=seed)   # drop agent >=2.5m away
    planner.reset(); planner.set_actions(NAV_SKILLS)
    instr = task["instruction"]

    img_path, fl = render_inject(env, target_id, patch, save_copy=f"{ep_dir}/frame00_obs.png")
    steps = success = 0
    reached = False
    dframe = 0                 # decision counter (for the per-step grounded view)
    frames_with_patch = int(fl)
    trace = []
    dist = env.nav_distance(target_id)
    dists = [dist]
    while steps < max_steps and not reached:
        if dist <= success_dist:             # EB-Navigation auto-terminate on proximity
            success, reached = 1, True
            trace.append((f"[within {success_dist}m]", True, f"dist={dist:.2f}")); break
        try:
            action, _reason = planner.act(img_path, instr)
        except Exception as e:
            import traceback as _tb; _tb.print_exc()
            trace.append(("PLANNER_ERROR", False, repr(e)[:100])); break
        # VINE's grounded view of the frame just planned on (nav planner writes obs_bbox.png)
        _bx = "output/rung2/obs/obs_bbox.png"
        if os.path.exists(_bx):
            try:
                import shutil; shutil.copy(_bx, f"{ep_dir}/dec{dframe:02d}_grounded.png")
            except Exception:
                pass
        dframe += 1
        plan = action if isinstance(action, list) else [action]
        for a in plan:
            if steps >= max_steps:
                break
            lang = a if isinstance(a, str) else (NAV_SKILLS[a] if isinstance(a, int)
                                                 and 0 <= a < len(NAV_SKILLS) else None)
            if lang is None:
                trace.append((str(a), False, "invalid action id")); steps += 1; continue
            ok, fb = env.nav_move(lang); steps += 1
            dist = env.nav_distance(target_id); dists.append(dist)
            success = 1 if dist <= success_dist else 0
            try:
                planner.update_info({"action_id": a if isinstance(a, int) else 0,
                                     "env_feedback": fb, "last_action_success": float(ok),
                                     "task_success": float(success)})
            except Exception:
                pass
            img_path, fl = render_inject(env, target_id, patch,
                                         save_copy=f"{ep_dir}/frame{steps:02d}_obs.png")
            frames_with_patch += int(fl)
            trace.append((lang, bool(ok), f"{fb}; dist={dist:.2f}"))
            if success:
                reached = True; break
    env.stop()
    result = {"success": bool(success), "reached": reached, "steps": steps,
              "start_dist": round(float(start_dist), 3) if start_dist is not None else None,
              "final_dist": round(float(dist), 3), "min_dist": round(float(min(dists)), 3),
              "frames_with_patch": frames_with_patch, "trace": trace}
    json.dump({"task": task["id"], "instruction": instr, "condition": tag, **result},
              open(f"{ep_dir}/episode.json", "w"), indent=2, default=str)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", default="rung2/gap_tasks_v2.json")
    ap.add_argument("--task", default="all", help="task id, comma list, or 'all'")
    ap.add_argument("--conditions", default="base_benign,esca_benign,esca_patch")
    ap.add_argument("--model", default="gpt-4o", help="planner LLM (e.g. gpt-4o, gpt-4o-mini)")
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7", help="comma list of start-pose seeds; SR = successes / len(seeds)")
    ap.add_argument("--success-dist", type=float, default=1.0,
                    help="GT distance (m) counted as reaching the target (EB-Navigation uses 1.0)")
    args = ap.parse_args()
    data = json.load(open(args.tasks_file))
    TASKS = {t["id"]: t for t in data["tasks"]}
    task_ids = list(TASKS) if args.task == "all" else args.task.split(",")
    missing = [t for t in task_ids if t not in TASKS]
    if missing:
        raise SystemExit(f"[nav] unknown task id(s) {missing}. Available: {list(TASKS)}")
    conds = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    os.makedirs("output/rung2/nav", exist_ok=True)
    report_path = "output/rung2/nav/report.json"

    cache = {}
    def get_planner(sgt, lang):
        if (sgt, lang) not in cache:
            print(f"  [loading nav planner {args.model} sg_text={sgt} language_only={lang}]", flush=True)
            cache[(sgt, lang)] = make_nav_planner(sgt, lang, args.model)
        return cache[(sgt, lang)]

    def _mean(v):
        v = [x for x in v if x is not None]
        return round(sum(v) / len(v), 3) if v else None

    summary = {}
    for tid in task_ids:
        task = TASKS[tid]
        print(f"\n### NAV {tid}: {task['instruction']}  ({task['scene']}, {task['target_type']})  "
              f"[{args.model}, {len(seeds)} seeds, success<={args.success_dist}m]", flush=True)
        summary[tid] = {"target": task["target_type"], "scene": task["scene"], "mode": task.get("mode")}
        for cond in conds:
            sgt, lang, inj = CONDS[cond]
            planner = get_planner(sgt, lang)
            succs, mins, finals, pfs = [], [], [], []
            for seed in seeds:
                try:
                    r = run_episode(task, planner, inj, f"{cond}__s{seed}", args.max_steps, seed, args.success_dist)
                except Exception as e:
                    import traceback as _tb; _tb.print_exc()
                    r = {"success": False, "min_dist": None, "final_dist": None, "steps": 0,
                         "frames_with_patch": 0, "error": repr(e)[:100]}
                succs.append(int(bool(r.get("success"))))
                mins.append(r.get("min_dist")); finals.append(r.get("final_dist")); pfs.append(r.get("frames_with_patch", 0))
                print(f"    {cond} s{seed}: {'S' if succs[-1] else 'F'}  min={r.get('min_dist')} "
                      f"final={r.get('final_dist')} steps={r.get('steps')} pf={r.get('frames_with_patch')}", flush=True)
            n = len(seeds)
            rec = {"sr": round(sum(succs) / n, 3), "succ": sum(succs), "n": n,
                   "min_mean": _mean(mins), "final_mean": _mean(finals),
                   "min_best": round(min([m for m in mins if m is not None], default=float("nan")), 3),
                   "patch_frames_mean": round(sum(pfs) / n, 1)}
            summary[tid][cond] = rec
            print(f"  {cond:14s} SR={rec['succ']}/{n}={rec['sr']:.0%}  min_mean={rec['min_mean']}m  final_mean={rec['final_mean']}m", flush=True)
            # incremental save so a long run is never lost
            json.dump({"model": args.model, "seeds": seeds, "success_dist": args.success_dist,
                       "max_steps": args.max_steps, "conditions": conds, "tasks": summary},
                      open(report_path, "w"), indent=2, default=str)

    # ---- comprehensive report ----
    print("\n" + "#" * 96)
    print(f"#  NAV MIM COMPREHENSIVE REPORT   model={args.model}   seeds={len(seeds)}   success<={args.success_dist}m   max_steps={args.max_steps}")
    print("#" * 96)
    for cond in conds:
        print(f"\n== {cond} ==")
        print("  " + "task".ljust(16) + "mode".ljust(11) + "SR".ljust(9) + "min_dist".ljust(11) + "final_dist")
        for tid in task_ids:
            rec = summary.get(tid, {}).get(cond)
            if rec:
                print("  " + tid.ljust(16) + str(summary[tid].get("mode")).ljust(11)
                      + f"{rec['succ']}/{rec['n']}".ljust(9)
                      + f"{rec['min_mean']}".ljust(11) + f"{rec['final_mean']}")
    if "esca_benign" in conds and "esca_patch" in conds:
        print("\n== ATTACK EFFECT  esca_benign -> esca_patch  (want SR down, distances up) ==")
        print("  " + "task".ljust(16) + "SR".ljust(16) + "min_dist".ljust(20) + "final_dist")
        for tid in task_ids:
            b, p = summary.get(tid, {}).get("esca_benign"), summary.get(tid, {}).get("esca_patch")
            if b and p:
                print("  " + tid.ljust(16)
                      + f"{b['succ']}/{b['n']} -> {p['succ']}/{p['n']}".ljust(16)
                      + f"{b['min_mean']} -> {p['min_mean']}".ljust(20)
                      + f"{b['final_mean']} -> {p['final_mean']}")
    print(f"\n(report saved to {report_path})")


if __name__ == "__main__":
    main()
