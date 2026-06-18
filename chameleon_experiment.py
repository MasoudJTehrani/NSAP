#!/usr/bin/env python
"""
Full per-scenario chameleon sweep across category / attribute / relation targets.

For each target:
  Phase 1 (loud):   train a universal patch on ALL its scenarios, NO chameleon.
  Phase 2 (hidden): for N scenarios, WARM-START that patch and fine-tune the
                    chameleon camouflage on ONLY that scenario (strong smoothing),
                    so it blends into that exact kitchen's object.

Reports, per target, the aggregate per-frame flip rate of the loud universal patch
vs the per-scenario chameleon patches. Saves hidden-patch sample images per scenario.

Run:  conda run -n NSAP python chameleon_experiment.py
"""

import os
import sys
import json
import random
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "SceneGraphBench/data/kitchen_pilot")
EXP  = os.path.join(ROOT, "output/chexp")

N         = 3
P1_STEPS  = 1500
P2_STEPS  = 600
LAMBDA_TV = "1e-2"     # strong smoothing
LAMBDA_C  = "3"        # strong camouflage

# (mode, target, attribute, relation)
SPECS = [
    ("category",  "pot",            "", ""),
    ("category",  "kettle",         "", ""),
    ("category",  "toaster",        "", ""),
    ("category",  "coffee machine", "", ""),
    ("category",  "fridge",         "", ""),
    ("category",  "microwave",      "", ""),
    ("attribute", "fridge",         "open", ""),
    ("attribute", "microwave",      "open", ""),
    ("attribute", "kettle",         "open", ""),
    ("relation",  "pot",            "", "on"),
    ("relation",  "toaster",        "", "on"),
    ("relation",  "coffee machine", "", "on"),
]

ENV = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
RUN = [sys.executable, os.path.join(ROOT, "run_attack.py")]
MAN = json.load(open(os.path.join(DATA, "manifest.json")))


def run(args):
    print("  $ run_attack " + " ".join(args), flush=True)
    subprocess.run(RUN + args, env=ENV, check=False)


def spec_args(spec):
    mode, target, attr, rel = spec
    a = ["--mode", mode, "--target", target]
    if mode == "attribute":
        a += ["--attribute", attr]
    if mode == "relation":
        a += ["--relation", rel]
    return a


def spec_slug(spec):
    mode, target, attr, rel = spec
    t = target.replace(" ", "_")
    if mode == "attribute":
        return f"attribute_{t}_{attr.replace(' ', '_')}"
    if mode == "relation":
        return f"relation_{rel.replace(' ', '_')}_{t}"
    return f"category_{t}"


def read_flip(patch_path):
    j = patch_path.replace("_final.pt", "_final_eval.json")
    if os.path.exists(j):
        d = json.load(open(j))
        return d.get("n_flipped") or 0, d.get("n_total") or 0
    return 0, 0


def pick_scenarios(spec, n):
    """Pick n scenarios of the target category; for attribute/relation prefer ones
    whose ground truth actually contains the target condition (so Phase 2 has
    clean instances instead of fast-failing on a wrong-state scene)."""
    mode, target, attr, rel = spec
    cands = [e["scenario_id"] for e in MAN["scenarios"] if e["target_category"] == target]
    random.Random(0).shuffle(cands)
    if mode == "category":
        return cands[:n]
    good = []
    for sid in cands:
        sg = json.load(open(os.path.join(DATA, "scenarios", sid, "scene_graph.json")))
        tid = sg["target"]["id"]
        hit = False
        for fr in sg["frames"]:
            if mode == "attribute":
                hit = any(o["id"] == tid and any(attr.lower() in a.lower() for a in o["attributes"])
                          for o in fr["objects"])
            else:
                hit = any(r["subject"] == tid and rel.lower() in r["predicate"].lower()
                          for r in fr["relations"])
            if hit:
                break
        if hit:
            good.append(sid)
        if len(good) >= n:
            break
    return good or cands[:n]


def main():
    summary = []
    for spec in SPECS:
        slug = spec_slug(spec)
        print(f"\n\n##################### {slug} #####################", flush=True)

        # Phase 1: universal (all data, no chameleon)
        uni_dir = os.path.join(EXP, f"uni_{slug}")
        run(spec_args(spec) + ["--sim-data", DATA, "--no-chameleon", "--batch", "2",
                               "--all-data", "--steps", str(P1_STEPS), "--output-dir", uni_dir])
        uni_patch = os.path.join(uni_dir, slug, f"{slug}_final.pt")
        if not os.path.exists(uni_patch):
            print(f"  ! {slug}: Phase-1 produced no patch (no clean instances); skipping.")
            summary.append((slug, 0, 0, 0, 0))
            continue

        scenarios = pick_scenarios(spec, N)
        print(f"  scenarios: {scenarios}", flush=True)
        loud_f = loud_t = hid_f = hid_t = 0
        for sid in scenarios:
            run(["--eval", "--patch", uni_patch, "--only-scenario", sid])
            lf, lt = read_flip(uni_patch)
            loud_f += lf; loud_t += lt

            cham_dir = os.path.join(EXP, f"cham_{slug}_{sid}")
            run(spec_args(spec) + ["--sim-data", DATA, "--batch", "1", "--init-patch", uni_patch,
                                   "--only-scenario", sid, "--steps", str(P2_STEPS),
                                   "--lambda-tv", LAMBDA_TV, "--lambda-c", LAMBDA_C,
                                   "--output-dir", cham_dir])
            cham_patch = os.path.join(cham_dir, slug, f"{slug}_final.pt")
            if os.path.exists(cham_patch):
                run(["--eval", "--patch", cham_patch, "--only-scenario", sid])
                hf, ht = read_flip(cham_patch)
                hid_f += hf; hid_t += ht

        summary.append((slug, loud_f, loud_t, hid_f, hid_t))
        print(f"  >> {slug}: loud {loud_f}/{loud_t}  hidden {hid_f}/{hid_t}", flush=True)

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n\n############### CHAMELEON SWEEP SUMMARY (per-frame flip) ###############")
    print(f"{'target':<28}{'no-chameleon':>16}{'chameleon':>16}")
    for slug, lf, lt, hf, ht in summary:
        lr = f"{lf/lt:.0%} ({lf}/{lt})" if lt else "n/a"
        hr = f"{hf/ht:.0%} ({hf}/{ht})" if ht else "n/a"
        print(f"{slug:<28}{lr:>16}{hr:>16}")
    json.dump([{"target": s, "loud_flipped": lf, "loud_total": lt,
                "hidden_flipped": hf, "hidden_total": ht}
               for s, lf, lt, hf, ht in summary],
              open(os.path.join(EXP, "sweep_summary.json"), "w"), indent=2)
    print(f"\nSaved {os.path.join(EXP, 'sweep_summary.json')}")


if __name__ == "__main__":
    main()
