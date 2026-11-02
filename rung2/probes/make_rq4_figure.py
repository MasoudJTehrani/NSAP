"""Build the RQ4 figure: the same patch optimised three ways, plus the deployed patch in scene.

Written with matplotlib so the labels stay vector in the PDF while the frames embed as
raster, which is what a conference template wants. Writes both a PDF and a 300 dpi PNG.

The in-scene column needs rung2/probes/render_patch_in_scene.py to have been run for each
row's tag first.

  python rung2/probes/make_rq4_figure.py --out output/probes/fig_rq4_arms
"""
import argparse, os, sys

NSAP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(NSAP)
import matplotlib
matplotlib.use("Agg")
# Type 3 fonts are rejected by the ACM and IEEE PDF checkers; 42 embeds TrueType instead.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from PIL import Image

# head label, patch directory name, scenario id, tag used by render_patch_in_scene.py
ROWS = [("Category\n(toaster)", "category_toaster", "pilot_scenario_00006", "toaster"),
        ("Attribute\n(television)", "attribute_Television_displaying", "tv_scenario_00000", "television"),
        ("Relation\n(pot)", "relation_on_Pot", "pot_scenario_00001", "pot")]

COLS = [("Standard patch", "output/ablation_patches/noinit"),
        ("Object initialization", "output/ablation_patches/content0"),
        ("Chameleon (deployed)", "output/chameleon_patches")]

SCENE_COL = "Chameleon in the scene"
SCENE_DIR = "output/probes/in_scene"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/probes/fig_rq4_arms")
    ap.add_argument("--width", type=float, default=7.0, help="figure width in inches")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    ncol = len(COLS) + 1
    nrow = len(ROWS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(a.width, a.width * nrow / ncol * 1.02))
    plt.rcParams["font.family"] = "DejaVu Sans"

    for ri, (rname, head, scen, tag) in enumerate(ROWS):
        paths = [f"{root}/{head}/{scen}/{head}/{head}_final_preview.png" for _, root in COLS]
        paths.append(f"{SCENE_DIR}/{tag}_patched.png")
        for ci, p in enumerate(paths):
            ax = axes[ri][ci]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.6")
            if os.path.exists(p):
                # nearest keeps the patch's own pixel structure visible rather than
                # smoothing it into something the optimisation never produced
                ax.imshow(Image.open(p).convert("RGB"), interpolation="nearest", aspect="equal")
            else:
                print(f"  missing: {p}")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
            if ri == 0:
                ax.set_title((COLS + [(SCENE_COL, None)])[ci][0], fontsize=8, pad=5)
            if ci == 0:
                ax.set_ylabel(rname, fontsize=8, labelpad=6)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.005, wspace=0.06, hspace=0.06)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".pdf", format="pdf", dpi=a.dpi, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(a.out + ".png", format="png", dpi=a.dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {a.out}.pdf and {a.out}.png")


if __name__ == "__main__":
    main()
