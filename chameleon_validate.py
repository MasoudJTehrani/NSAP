#!/usr/bin/env python
"""
Validate the NEW appearance camouflage (object-init + L_content anchor) on ONE
fridge scenario, sweeping the stealth<->flip knob lambda_content.

For each lambda_content: object-init the patch from the real fridge crop (reusing
the loud universal patch's anchor/size), train content-only (no gram/tv/dino) to
isolate the new mechanism, then eval the per-frame flip rate on that scenario.

Baselines on the same scenario: the loud universal patch, and the OLD (gram) chameleon.
Prints flip rate per variant; the patch previews live next to each .pt for eyeballing.

Run:  conda run -n NSAP python chameleon_validate.py
"""
import os, sys, json, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "SceneGraphBench/data/kitchen_pilot")
SID  = "scenario_00081"
LOUD = os.path.join(ROOT, "output/chexp/uni_category_fridge/category_fridge/category_fridge_final.pt")
OLD_CHAM = os.path.join(ROOT, "output/chexp/cham_category_fridge_scenario_00081/category_fridge/category_fridge_final.pt")
OUT  = os.path.join(ROOT, "output/chexp_val")
ENV  = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
RUN  = [sys.executable, os.path.join(ROOT, "run_attack.py")]

LAMBDAS = [2.0, 10.0, 50.0]
STEPS   = 600


def run(args):
    print("  $ run_attack " + " ".join(str(a) for a in args), flush=True)
    subprocess.run(RUN + [str(a) for a in args], env=ENV, check=False)


def flip_of(patch_pt):
    j = patch_pt.replace("_final.pt", "_final_eval.json")
    if os.path.exists(j):
        d = json.load(open(j))
        return d.get("n_flipped") or 0, d.get("n_total") or 0
    return 0, 0


def eval_patch(patch_pt):
    run(["--eval", "--patch", patch_pt, "--only-scenario", SID])
    return flip_of(patch_pt)


def main():
    rows = []

    # Baselines
    print("\n##### baseline: LOUD universal #####", flush=True)
    rows.append(("loud_universal", LOUD, *eval_patch(LOUD)))
    if os.path.exists(OLD_CHAM):
        print("\n##### baseline: OLD gram-chameleon #####", flush=True)
        rows.append(("old_gram_chameleon", OLD_CHAM, *eval_patch(OLD_CHAM)))

    # New: object-init + content sweep
    for lc in LAMBDAS:
        slug = f"objinit_lc{lc:g}"
        odir = os.path.join(OUT, slug)
        patch_pt = os.path.join(odir, "category_fridge", "category_fridge_final.pt")
        print(f"\n##### {slug} #####", flush=True)
        if not os.path.exists(patch_pt):
            run(["--mode", "category", "--target", "fridge", "--sim-data", DATA,
                 "--init-patch", LOUD, "--object-init", "--no-chameleon",
                 "--lambda-content", lc, "--only-scenario", SID,
                 "--coverage", 0.5, "--eot-crops", 4, "--batch", 1,
                 "--steps", STEPS, "--output-dir", odir])
        if os.path.exists(patch_pt):
            rows.append((slug, patch_pt, *eval_patch(patch_pt)))
        else:
            rows.append((slug, patch_pt, 0, 0))

    print("\n\n############### CHAMELEON VALIDATION (scenario_00081, fridge) ###############")
    print(f"{'variant':<22}{'flip rate':>14}   preview")
    for name, pt, f, t in rows:
        rate = f"{f/t:.0%} ({f}/{t})" if t else "n/a"
        png = pt.replace("_final.pt", "_final_preview.png")
        print(f"{name:<22}{rate:>14}   {png}")
    json.dump([{"variant": n, "patch": p, "flipped": f, "total": t} for n, p, f, t in rows],
              open(os.path.join(OUT, "validate_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
