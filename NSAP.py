import os
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

    print("\n=== Step 2: Downloading videos (500 videos via yt-dlp) ===")
    download_videos(max_videos=500)

    print("\n=== Step 3: Extracting frames ===")
    extract_frames()

    print("\n=== Step 4: Downloading model ===")
    download_model()

    print("\n=== Step 5: Downloading SAM2 checkpoint ===")
    download_sam2()


if not all((DATASET_DIR / f).exists() for f in ESCA_FILES):
    print("Prerequisites not found. Starting setup...")
    download_prerequisites()
else:
    print("Prerequisites found.")
