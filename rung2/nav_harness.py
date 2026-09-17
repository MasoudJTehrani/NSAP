#!/usr/bin/env python
"""Navigation MIM planner factory: the fixed ESCA EBNavigationPlanner wired to VINE's
trained SGClip weights, shared by the two faithful benchmark runners (eb_bench_runner.py
for category, staged_bench.py for attribute/relation). The standalone custom nav loop that
used to live here was removed once those benchmarks became the canonical runners.

Run in the `embench` env.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # rung2/
import harness   # noqa: F401  -- sets sys.path (ESCA/LASER), OPENAI/GD/VINE env vars, os.chdir(NSAP)
from env import Rung2Env                                          # noqa: E402

from embodiedbench.planner.navigation.esca import EBNavigationPlanner                 # noqa: E402
from embodiedbench.evaluator.config.system_prompts import eb_laser_navigation_system_prompt  # noqa: E402
from embodiedbench.evaluator.config.eb_navigation_example import (                    # noqa: E402
    examples as NAV_EX, laser_sg_examples_no_revert as NAV_SG_EX, target_examples as NAV_TGT_EX)

NAV_SKILLS = Rung2Env.NAV_SKILLS


_FAITHFUL = os.environ.get("NSAP_ESCA_FAITHFUL", "0") == "1"
# gd_only is decoupled so it can be varied while every other parameter stays faithful.
# True  = ESCA's shipped nav config: predict_laser is skipped, SGClip never runs, and the
#         detector query is built as a STRING, which predict_gd joins character-wise.
# False = SGClip is in the loop and the detector gets a proper class list.
_GD_ONLY = os.environ.get("NSAP_GD_ONLY", "1" if _FAITHFUL else "0") == "1"


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
        # ESCA-FAITHFUL MODE. evaluator/navigation/esca.py:305-325 passes a specific config
        # that our harness did not match. NSAP_ESCA_FAITHFUL=1 reproduces it exactly:
        #   n_shots 3, chat_history True, multistep True, detection_box True, gd_only True,
        #   tp 4, top_k 1, aggr_thres 0.3
        # NOTE gd_only=True SKIPS predict_laser entirely (planner/navigation/esca.py:759),
        # so SGClip is never called and the scene graph is GroundingDINO labels + boxes.
        n_shot=(3 if _FAITHFUL else 2), obs_key="head_rgb",
        chat_history=(True if _FAITHFUL else False), language_only=language_only,
        multistep=(True if _FAITHFUL else False),
        scene_graph_text=scene_graph_text,
        gd_only=_GD_ONLY,
        tp=(4 if _FAITHFUL else 1),
        # ESCA ships top_k=1 (planner/navigation/esca.py:177), i.e. the gate admits exactly
        # ONE object into the scene graph. Keep that as the default so deployment runs stay
        # faithful; NSAP_TOP_K lets an evaluation widen it (e.g. 15, as the SGG attack
        # literature evaluates over) so a multi-object graph exists to measure isolation on.
        top_k=int(os.environ.get("NSAP_TOP_K", "1")),
        aggr_thres=float(os.environ.get("NSAP_AGGR_THRES", "0.3" if _FAITHFUL else "0.4")),
        # CRITICAL: boundingbox gates whether grounding survives -> whether the scene graph
        # is actually built into the prompt (with it False the planner resets target_labels=[]
        # and NO SG ever reaches the MLLM). Tie it to scene_graph_text so esca/lang truly use
        # VINE's SG (and base doesn't). Also makes the planner write the per-step grounded view.
        boundingbox=(True if _FAITHFUL else scene_graph_text),
        gd_config_path=gd_cfg, gd_checkpoint_path=os.environ["NSAP_GD_CKPT"])
