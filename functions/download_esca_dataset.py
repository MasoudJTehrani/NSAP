"""
Download the ESCA-video-87K dataset (LASER project) from HuggingFace into the dataset folder.

  Annotations : gpt_specs1_id_LLaVA_all.jsonl  (~211 MB)
                gpt_specs2_id_LLaVA_all.jsonl  (~407 MB)
                neg_examples.jsonl             (~1.91 GB)
  Mask subset : mini_LLaVA_10000_masks.tar.gz  (~2.28 GB, the 10k-clip subset NSAP uses)
  Mask chunks : chunk_000.tar.gz ... chunk_043.tar.gz (~33 GB) — the FULL ~87K-clip
                mask set. These are MASKS, not videos (confirmed: each chunk is
                thousands of *_mask.pkl files). The dataset contains NO videos at
                all; videos must be fetched from YouTube via download_videos.py.

Source: https://huggingface.co/datasets/video-fm/ESCA-video-87K
"""

import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

MASKS_SENTINEL = DATASET_DIR / ".masks_extracted"

REPO_ID = "video-fm/ESCA-video-87K"

ANNOTATION_FILES = [
    "gpt_specs1_id_LLaVA_all.jsonl",
    "gpt_specs2_id_LLaVA_all.jsonl",
    "neg_examples.jsonl",
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


def _extract(archive: Path, cleanup: bool = False) -> None:
    print(f"  Extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for i, member in enumerate(members, 1):
            tf.extract(member, DATASET_DIR, filter="data")
            if i % 2000 == 0 or i == len(members):
                pct = i / len(members) * 100
                print(f"\r  {pct:5.1f}%  ({i}/{len(members)} files)", end="", flush=True)
    print(f"\r  Done: {len(members)} files extracted from {archive.name}.")
    if cleanup:
        archive.unlink()
        print(f"  Deleted {archive.name} to save space.")


def main(download_chunks: bool = False, cleanup: bool = False) -> None:
    """
    download_chunks: download all 44 mask-chunk archives (~33 GB compressed) — the
                     FULL ~87K-clip mask set. These are MASKS, not videos. (Videos
                     are not part of this dataset; fetch them from YouTube with
                     download_videos.py.)
    cleanup:         delete archives after extraction to save space.
    """
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
    if MASKS_SENTINEL.exists():
        print("  Already extracted — skipping.")
    else:
        archive = _download_file(hf, "mini_LLaVA_10000_masks.tar.gz")
        _extract(archive, cleanup=cleanup)
        MASKS_SENTINEL.touch()

    if download_chunks:
        print("\n=== Full mask set: 44 chunks (~33 GB, ~87K clips' masks — NOT videos) ===")
        for fname in [f"chunk_{i:03d}.tar.gz" for i in range(44)]:
            archive = _download_file(hf, fname)
            _extract(archive, cleanup=cleanup)
    else:
        print("\nSkipping the full mask chunks (~33 GB); using the mini 10k mask subset.")
        print("Note: videos are NOT in this dataset — fetch them from YouTube via download_videos.py.")

    print("\nAll done. Dataset is in:", DATASET_DIR)


if __name__ == "__main__":
    include_chunks = "--chunks" in sys.argv
    do_cleanup = "--cleanup" in sys.argv
    try:
        main(download_chunks=include_chunks, cleanup=do_cleanup)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
