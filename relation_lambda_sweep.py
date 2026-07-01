#!/usr/bin/env python
"""
Relation attack: sweep lambda_content DOWN to recover attack strength.

Hypothesis: at lambda_content=50 the appearance anchor (keep the subject looking
like itself, in place) preserves the very "on" cue the relation attack must destroy,
so it gutted the attack (on_pot 58->14%). A WEAKER anchor should let the patch drift
enough to break the relation, at some stealth cost.

For each relation target, on one representative scenario, train object-init+content at
lambda in {5,10,20} (no TV) and eval per-frame flip, vs the loud universal and the
existing lambda=50 patch from the rollout.

Run: conda run -n NSAP python relation_lambda_sweep.py
"""
import os, sys, json, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "SceneGraphBench/data/kitchen_pilot")
ENV  = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
RUN  = [sys.executable, os.path.join(ROOT, "run_attack.py")]
LAMBDAS = [5, 10, 20]
STEPS   = 600
OUT     = os.path.join(ROOT, "output/relsweep")

# (slug, --target, --relation, representative scenario)
REL = [
    ("relation_on_pot",            "pot",            "on", "scenario_00133"),
    ("relation_on_toaster",        "toaster",        "on", "scenario_00086"),
    ("relation_on_coffee_machine", "coffee machine", "on", "scenario_00079"),
]


def run(a):
    print("  $ run_attack " + " ".join(map(str, a)), flush=True)
    subprocess.run(RUN + list(map(str, a)), env=ENV, check=False)


def flip(pt):
    j = pt.replace("_final.pt", "_final_eval.json")
    if os.path.exists(j):
        d = json.load(open(j)); return d.get("n_flipped") or 0, d.get("n_total") or 0
    return 0, 0


def ev(pt, sid):
    run(["--eval", "--patch", pt, "--only-scenario", sid]); return flip(pt)


def main():
    rows = []
    for slug, tgt, rel, sid in REL:
        print(f"\n##################### {slug} / {sid} #####################", flush=True)
        loud = os.path.join(ROOT, f"output/chexp/uni_{slug}/{slug}/{slug}_final.pt")
        loud_ft = ev(loud, sid)                                   # loud baseline on this scenario
        p50 = os.path.join(ROOT, f"output/chexp/cham_{slug}_{sid}/{slug}/{slug}_final.pt")
        p50_ft = flip(p50) if os.path.exists(p50) else (0, 0)     # rollout lambda=50

        res = {}
        for lc in LAMBDAS:
            od = os.path.join(OUT, f"lc{lc}_{slug}_{sid}")
            pt = os.path.join(od, slug, f"{slug}_final.pt")
            if not os.path.exists(pt):
                run(["--mode", "relation", "--target", tgt, "--relation", rel, "--sim-data", DATA,
                     "--init-patch", loud, "--object-init", "--no-chameleon", "--lambda-content", lc,
                     "--lambda-tv", "0", "--only-scenario", sid, "--coverage", "0.5", "--eot-crops", "4",
                     "--batch", "1", "--steps", STEPS, "--output-dir", od])
            res[lc] = ev(pt, sid) if os.path.exists(pt) else (0, 0)
        rows.append((slug, sid, loud_ft, p50_ft, res))

    def fr(ft):
        f, t = ft; return f"{f}/{t}" if t else "n/a"
    print("\n\n############ RELATION lambda_content sweep (per-scenario flip) ############")
    print(f"{'target':<28}{'scenario':<16}{'loud':>9}{'λ=5':>9}{'λ=10':>9}{'λ=20':>9}{'λ=50':>9}")
    for slug, sid, loud_ft, p50_ft, res in rows:
        print(f"{slug:<28}{sid:<16}{fr(loud_ft):>9}{fr(res[5]):>9}{fr(res[10]):>9}{fr(res[20]):>9}{fr(p50_ft):>9}")
    json.dump([{"target": s, "scenario": sid, "loud": l, "lambda50": p,
                "lambda5": r[5], "lambda10": r[10], "lambda20": r[20]}
               for s, sid, l, p, r in rows],
              open(os.path.join(OUT, "relsweep_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
