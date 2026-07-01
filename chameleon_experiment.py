#!/usr/bin/env python
"""
Full per-scenario chameleon sweep across category / attribute / relation targets.

For each target:
  Phase 1 (loud):   train a universal patch on ALL its scenarios, NO chameleon.
  Phase 2 (hidden): for N scenarios, OBJECT-INIT the patch from that scenario's real
                    target crop (the object's appearance under the patch footprint) and
                    re-optimise the attack under a pixel content anchor (lambda_content),
                    so the patch keeps LOOKING like the object while still flipping it.
                    Replaces the old gram/style camouflage, which only matched texture
                    statistics and so produced a loud rainbow.

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
LAMBDA_CONTENT = "50"  # object-init appearance anchor (proven: 100% flip + hidden at cov 0.5)
EOT_CROPS = "4"        # Fix-B: jittered detector-like crops per step (live-loop transfer)
COVERAGE  = "0.5"      # Fix-B: bigger patch so it dominates GroundingDINO's boxes
# TV off: it costs a flip (erases the high-freq the attack needs) and leaks color into
# an iridescent sheen. The content anchor alone keeps the patch looking like the object.
FIXB      = ["--eot-crops", EOT_CROPS, "--coverage", COVERAGE, "--lambda-tv", "0"]

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
    # kettle "open" dropped: the kettle is tiny and its lid open/closed state is
    # ambiguous and irrelevant — the attack was meaningless (0% flip). Use fridge + microwave.
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

        # Phase 1: universal (all data, no chameleon) — skip if already trained (resume)
        uni_dir = os.path.join(EXP, f"uni_{slug}")
        uni_patch = os.path.join(uni_dir, slug, f"{slug}_final.pt")
        if os.path.exists(uni_patch):
            print(f"  (resume) reusing universal patch {uni_patch}")
        else:
            run(spec_args(spec) + ["--sim-data", DATA, "--no-chameleon", "--batch", "1",
                                   "--all-data", "--steps", str(P1_STEPS), "--output-dir", uni_dir] + FIXB)
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
            cham_patch = os.path.join(cham_dir, slug, f"{slug}_final.pt")
            if not os.path.exists(cham_patch):   # skip training if already done (resume)
                run(spec_args(spec) + ["--sim-data", DATA, "--batch", "1", "--init-patch", uni_patch,
                                       "--object-init", "--no-chameleon", "--lambda-content", LAMBDA_CONTENT,
                                       "--only-scenario", sid, "--steps", str(P2_STEPS),
                                       "--output-dir", cham_dir] + FIXB)
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
