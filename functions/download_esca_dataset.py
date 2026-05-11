"""
Download the ESCA-video-87K dataset (LASER project) from HuggingFace into the dataset folder.

  Annotations : gpt_specs1_id_LLaVA_all.jsonl  (~211 MB)
                gpt_specs2_id_LLaVA_all.jsonl  (~407 MB)
                neg_examples.jsonl             (~1.91 GB)
  Mask subset : mini_LLaVA_10000_masks.tar.gz  (~2.28 GB)
  Video chunks: chunk_000.tar.gz ... chunk_043.tar.gz (~33 GB)

Source: https://huggingface.co/datasets/video-fm/ESCA-video-87K
"""

import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "video-fm/ESCA-video-87K"

ANNOTATION_FILES = [
    "gpt_specs1_id_LLaVA_all.jsonl",
    "gpt_specs2_id_LLaVA_all.jsonl",
    "neg_examples.jsonl",
]

ARCHIVE_FILES = [
    "mini_LLaVA_10000_masks.tar.gz",
    *[f"chunk_{i:03d}.tar.gz" for i in range(44)],
]


HF_TOKEN = os.environ.get("HF_TOKEN")


def _require_hf_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface-hub")
        sys.exit(1)


def _download_file(hf, filename: str) -> Path:
    dest = DATASET_DIR / filename
    if dest.exists():
        print(f"  Already present: {filename} — skipping.")
        return dest
    print(f"  Downloading {filename} ...")
    path = hf.hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(DATASET_DIR),
        token=HF_TOKEN,
    )
    print(f"  Saved: {filename}")
    return Path(path)


def _extract(archive: Path) -> None:
    print(f"  Extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for i, member in enumerate(members, 1):
            tf.extract(member, DATASET_DIR, filter="data")
            if i % 2000 == 0 or i == len(members):
                pct = i / len(members) * 100
                print(f"\r  {pct:5.1f}%  ({i}/{len(members)} files)", end="", flush=True)
    print(f"\r  Done: {len(members)} files extracted from {archive.name}.")


def main(download_videos: bool = False, extract_masks: bool = False) -> None:
    hf = _require_hf_hub()
    if HF_TOKEN:
        print("HF_TOKEN found — using authenticated requests.")
    else:
        print("Warning: HF_TOKEN not set. Set it for higher rate limits and faster downloads.")
        print("  export HF_TOKEN=your_token_here\n")
    print(f"Dataset directory: {DATASET_DIR}\n")

    print("=== Annotation files ===")
    for fname in ANNOTATION_FILES:
        _download_file(hf, fname)

    print("\n=== Mask subset ===")
    archive = _download_file(hf, "mini_LLaVA_10000_masks.tar.gz")
    if extract_masks:
        _extract(archive)
        answer = input(f"\n  Delete {archive.name} to save space? [y/N] ").strip().lower()
        if answer == "y":
            archive.unlink()
            print(f"  Deleted {archive.name}")
    else:
        print("  Skipping extraction (pass extract_masks=True or --extract-masks flag to extract).")

    if download_videos:
        print("\n=== Video chunks (44 files, ~33 GB) ===")
        for fname in [f"chunk_{i:03d}.tar.gz" for i in range(44)]:
            archive = _download_file(hf, fname)
            _extract(archive)
            answer = input(f"\n  Delete {archive.name}? [y/N] ").strip().lower()
            if answer == "y":
                archive.unlink()
                print(f"  Deleted {fname}")
    else:
        print("\nSkipping video chunks (run with --videos flag to include them).")

    print("\nAll done. Dataset is in:", DATASET_DIR)


if __name__ == "__main__":
    include_videos = "--videos" in sys.argv
    include_masks = "--extract-masks" in sys.argv
    try:
        main(download_videos=include_videos, extract_masks=include_masks)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
