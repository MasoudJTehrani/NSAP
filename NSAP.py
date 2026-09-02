"""
NSAP setup entry point.

Downloads the prerequisites the attack/eval pipeline needs:
  - VINE model weights + source (model/vine/, src/LASER/)   [download_model]
  - SAM2 checkpoint (patch-anchor sub-region selection)     [download_sam2]

Optionally (--with-detection) it also fetches the GroundingDINO checkpoint. That
is NOT needed to TRAIN the attack (the SceneGraphBench sim data ships ground-truth
masks, so no detector frontend is required), but the closed-loop rung2 harness
grounds objects with GroundingDINO, so a replicator who runs the closed-loop
benchmark needs it.

The training/eval DATA is the self-contained SceneGraphBench (AI2-THOR) sim
dataset. Generate it from the SceneGraphBench/ submodule, or download the frozen
snapshot used in the paper, then point at it with `--sim-data <dir>` on
run_attack.py.

Usage:
    python NSAP.py                    # fetch model + SAM2
    python NSAP.py --with-detection   # also fetch GroundingDINO (for the harness)
"""

import argparse
from pathlib import Path

ROOT              = Path(__file__).resolve().parent
MODEL_DIR         = ROOT / "model" / "vine"
MODEL_SAFETENSORS = MODEL_DIR / "model.safetensors"
MODEL_CONFIG      = MODEL_DIR / "config.json"
SAM2_CHECKPOINT   = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
GDINO_CHECKPOINT  = ROOT / "checkpoints" / "groundingdino_swint_ogc.pth"
LASER_DIR         = ROOT / "src" / "LASER"

_NEXT_STEP = ("  python run_attack.py --sim-data <dataset> "
              "--mode category --target Toaster --steps 50")


def download_prerequisites(include_detection: bool = False) -> None:
    """Fetch the model + source (idempotent), SAM2, and optionally GroundingDINO."""
    from functions.download_model import main as download_model
    from functions.download_model import download_sam2

    print("=== Step 1: VINE model weights + source (model/vine/, src/LASER/) ===")
    download_model(include_detection=include_detection)

    print("\n=== Step 2: SAM2 checkpoint ===")
    download_sam2()


def _valid_file(path: Path, min_bytes: int) -> bool:
    """Exists and is at least min_bytes -- guards against half-written / pointer files."""
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _prereqs_present(include_detection: bool = False) -> bool:
    """A stricter check than "the file name exists": a partial download (weights
    present but the rest of the snapshot missing) must NOT read as complete."""
    ok = (_valid_file(MODEL_SAFETENSORS, 100_000_000)   # real VINE/CLIP weights are >100 MB
          and MODEL_CONFIG.is_file()                    # the HF snapshot actually finished
          and _valid_file(SAM2_CHECKPOINT, 10_000_000)
          and LASER_DIR.is_dir() and any(LASER_DIR.iterdir()))
    if include_detection:
        ok = ok and _valid_file(GDINO_CHECKPOINT, 100_000_000)
    return ok


def main():
    # RawDescriptionHelpFormatter so the module docstring renders verbatim in --help.
    ap = argparse.ArgumentParser(
        description="NSAP setup -- download the model + SAM2 (and optionally GroundingDINO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--with-detection", action="store_true",
                    help="also download the GroundingDINO checkpoint (needed for the rung2 closed-loop harness)")
    ap.add_argument("--with-dataset", action="store_true",
                    help="also download the frozen SceneGraphBench sim dataset snapshot from HuggingFace")
    args = ap.parse_args()

    if _prereqs_present(args.with_detection):
        print("Prerequisites already present:")
        print(f"  VINE weights : {MODEL_SAFETENSORS}")
        print(f"  SAM2 ckpt    : {SAM2_CHECKPOINT}")
        print(f"  LASER source : {LASER_DIR}")
        if args.with_detection:
            print(f"  GroundingDINO: {GDINO_CHECKPOINT}")
    else:
        print("Prerequisites not found or incomplete. Starting setup...")
        download_prerequisites(include_detection=args.with_detection)
        print("\nSetup complete.")

    if args.with_dataset:
        print("\n=== Sim dataset (frozen SceneGraphBench snapshot) ===")
        from functions.download_model import download_dataset
        download_dataset()

    print("\nNext: point at a SceneGraphBench dataset and run, e.g.")
    print(_NEXT_STEP)


if __name__ == "__main__":
    main()
