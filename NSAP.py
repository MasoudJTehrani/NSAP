"""
NSAP setup entry point.

Downloads the prerequisites the attack/eval pipeline needs:
  - VINE model weights + source (model/vine/, src/LASER/)   [download_model]
  - SAM2 checkpoint (patch-anchor sub-region selection)     [download_sam2]

The training/eval DATA is the self-contained SceneGraphBench (AI2-THOR) sim
dataset — generated separately (see SceneGraphBench/) and pointed at with
`--sim-data <dir>` on run_attack.py. It is NOT downloaded here.

Usage:
    python NSAP.py            # first-time setup: fetch model + SAM2
"""

import argparse
from pathlib import Path

ROOT              = Path(__file__).resolve().parent
MODEL_SAFETENSORS = ROOT / "model" / "vine" / "model.safetensors"
SAM2_CHECKPOINT   = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
LASER_DIR         = ROOT / "src" / "LASER"

_NEXT_STEP = ("  python run_attack.py --sim-data <dataset> "
              "--mode category --target Fridge --steps 50")


def download_prerequisites():
    """Fetch the model + source (idempotent) and the SAM2 checkpoint."""
    from functions.download_model import main as download_model
    from functions.download_model import download_sam2

    print("=== Step 1: VINE model weights + source (model/vine/, src/LASER/) ===")
    download_model()

    print("\n=== Step 2: SAM2 checkpoint ===")
    download_sam2()


def _prereqs_present() -> bool:
    return MODEL_SAFETENSORS.exists() and SAM2_CHECKPOINT.exists() and LASER_DIR.exists()


def main():
    # RawDescriptionHelpFormatter so the module docstring renders verbatim in --help.
    argparse.ArgumentParser(
        description="NSAP setup — download the model + SAM2 prerequisites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    ).parse_args()

    if _prereqs_present():
        print("Prerequisites already present:")
        print(f"  VINE weights : {MODEL_SAFETENSORS}")
        print(f"  SAM2 ckpt    : {SAM2_CHECKPOINT}")
        print(f"  LASER source : {LASER_DIR}")
    else:
        print("Prerequisites not found. Starting setup...")
        download_prerequisites()
        print("\nSetup complete.")

    print("\nNext: point at a SceneGraphBench dataset and run, e.g.")
    print(_NEXT_STEP)


if __name__ == "__main__":
    main()
