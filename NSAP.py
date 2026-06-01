"""
NSAP setup and utility entry point.

Usage
-----
# First-time setup (downloads ~500 videos, masks, model):
python NSAP.py

# Download more videos targeted at specific attack targets:
python NSAP.py --download-more --category person --attribute standing
python NSAP.py --download-more --category person --relation wearing
python NSAP.py --download-more --category person   # all person clips
python NSAP.py --download-more --category person --attribute standing --max 500

# Multiple targets in one run (comma-separated):
python NSAP.py --download-more --category person --attribute "standing,walking,running"
"""

import os
import argparse
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

ESCA_FILES = [
    "gpt_specs1_id_LLaVA_all.jsonl",
    "gpt_specs2_id_LLaVA_all.jsonl",
    "neg_examples.jsonl",
    ".masks_extracted",
    ".videos_downloaded",
    ".frames_extracted",
    ".model_downloaded",
    ".sam2_downloaded",
]


def download_prerequisites():
    from functions.download_esca_dataset import main as download_dataset
    from functions.download_videos import main as download_videos
    from functions.extract_frames import main as extract_frames
    from functions.download_model import main as download_model
    from functions.download_model import download_sam2

    print("=== Step 1: Downloading dataset (annotations + masks) ===")
    download_dataset()

    print("\n=== Step 2: Registering video download (on-demand per attack) ===")
    download_videos()   # touches sentinel only — actual downloads happen per attack

    print("\n=== Step 3: Extracting frames ===")
    extract_frames()

    print("\n=== Step 4: Downloading model ===")
    download_model()

    print("\n=== Step 5: Downloading SAM2 checkpoint ===")
    download_sam2()


def download_more_videos(
    category: str = '',
    attribute: str = '',
    relation: str = '',
    max_per_target: int = 300,
):
    """
    Download additional videos filtered by attack target.

    Supports comma-separated values for attribute/relation to download
    multiple targets in one call, e.g. attribute="standing,walking".
    """
    from functions.download_videos import download_for_targets

    # Build target list — split comma-separated values into separate targets
    targets = []
    attrs    = [a.strip() for a in attribute.split(',') if a.strip()] if attribute else ['']
    rels     = [r.strip() for r in relation.split(',')  if r.strip()] if relation  else ['']

    for attr in attrs:
        for rel in rels:
            t = {}
            if category:  t['category']  = category
            if attr:      t['attribute'] = attr
            if rel:       t['relation']  = rel
            if t:
                targets.append(t)

    if not targets:
        print("No targets specified. Use --category / --attribute / --relation.")
        return

    label_parts = []
    if category:  label_parts.append(f"category='{category}'")
    if attribute: label_parts.append(f"attribute='{attribute}'")
    if relation:  label_parts.append(f"relation='{relation}'")
    print(f"\n=== Downloading videos for {', '.join(label_parts)} (max {max_per_target} per target) ===")

    download_for_targets(targets, max_per_target=max_per_target)


def _show_dataset_stats():
    """Quick summary of what's available on disk."""
    videos_dir = DATASET_DIR / "videos"
    masks_dir  = DATASET_DIR / "mini_LLaVA_10000_masks"

    n_videos = len(list(videos_dir.glob("*.mp4"))) if videos_dir.exists() else 0
    n_masks  = len(list(masks_dir.rglob("*_mask.pkl"))) if masks_dir.exists() else 0

    # Usable = has both video and mask
    if videos_dir.exists() and masks_dir.exists():
        video_stems = {
            f.stem for f in videos_dir.iterdir()
            if f.suffix in (".mp4", ".webm", ".mkv", ".avi", ".mov")
        }
        mask_stems = {
            f.stem.removesuffix("_mask")
            for f in masks_dir.rglob("*_mask.pkl")
        }
        n_usable = len(video_stems & mask_stems)
    else:
        n_usable = 0

    print(f"  Videos on disk : {n_videos}")
    print(f"  Masks on disk  : {n_masks}")
    print(f"  Usable clips   : {n_usable}  (video + mask)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description="NSAP setup and dataset utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--download-more", action="store_true",
        help="Download additional videos (targeted or bulk)",
    )
    p.add_argument("--category",  default='', help="Target category  (e.g. 'person')")
    p.add_argument("--attribute", default='', help="Target attribute (e.g. 'standing', or 'standing,walking')")
    p.add_argument("--relation",  default='', help="Target relation  (e.g. 'wearing')")
    p.add_argument(
        "--max", type=int, default=300, dest="max_per_target",
        help="Max videos to download per target (default 300)",
    )
    p.add_argument("--stats", action="store_true", help="Show dataset stats and exit")
    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.stats:
        print("=== Dataset stats ===")
        _show_dataset_stats()
        return

    if args.download_more:
        if not all((DATASET_DIR / f).exists() for f in ESCA_FILES):
            print("Prerequisites not found — run `python NSAP.py` first to complete setup.")
            return
        download_more_videos(
            category      = args.category,
            attribute     = args.attribute,
            relation      = args.relation,
            max_per_target = args.max_per_target,
        )
        print("\n=== Updated dataset stats ===")
        _show_dataset_stats()
        return

    # Default: first-time setup
    if not all((DATASET_DIR / f).exists() for f in ESCA_FILES):
        print("Prerequisites not found. Starting setup...")
        download_prerequisites()
    else:
        print("Prerequisites found.")
        print("\n=== Dataset stats ===")
        _show_dataset_stats()
        print("\nTo download more videos run:  python NSAP.py --download-more --category person --attribute standing")


if __name__ == "__main__":
    main()
