import os
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
ESCA_FILES = [
    "gpt_specs1_id_LLaVA_all.jsonl",
    "gpt_specs2_id_LLaVA_all.jsonl",
    "neg_examples.jsonl",
]


def download_prerequisites():
    from functions.download_esca_dataset import main as download_dataset
    # from functions.extract_frames import main as extract_frames  # TODO: add when ready
    # from functions.download_model import main as download_model   # TODO: add when ready

    print("=== Step 1: Downloading dataset ===")
    download_dataset()

    print("\n=== Step 2: Extracting frames ===")
    # extract_frames()  # TODO: uncomment when ready

    print("\n=== Step 3: Downloading model ===")
    # download_model()  # TODO: uncomment when ready


if not all((DATASET_DIR / f).exists() for f in ESCA_FILES):
    print("Prerequisites not found. Starting setup...")
    download_prerequisites()
else:
    print("Prerequisites found.")
