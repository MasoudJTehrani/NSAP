"""
Download the VINE model (video-fm/vine) — the public release of SGCLIP/LASER.

Downloads:
  model/vine/    — HuggingFace snapshot (weights + vine source files, ~1.82 GB)
  src/LASER/     — LASER training code (PredicateModel, Scallop pipeline)

The vine source files (vine_model.py, vine_pipeline.py, vine_config.py) ship
directly inside the HF repo, so no separate vine_hf package clone is needed.
They are registered via a .pth file so they are importable as top-level modules.

GroundingDINO + SAM2 are NOT downloaded by default — we use pre-computed masks
from the dataset, so the detection frontend is not needed for adversarial attacks.
Pass include_detection=True to download them anyway.
"""

import os
import site
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model" / "vine"
SRC_DIR = ROOT / "src"
DATASET_DIR = ROOT / "dataset"
MODEL_SENTINEL = DATASET_DIR / ".model_downloaded"

HF_VINE_ID = "video-fm/vine"
HF_GDINO_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
HF_SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

REPOS = {
    "LASER": "https://github.com/video-fm/LASER.git",
}

HF_TOKEN = os.environ.get("HF_TOKEN")


def _require_hf_hub():
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install huggingface-hub")
        sys.exit(1)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Command failed: {' '.join(cmd)}")
        print(result.stderr[-2000:] if result.stderr else "(no stderr)")
        sys.exit(1)


def _download_vine_weights(hf) -> None:
    if MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors").exists():
        print("  Already downloaded — skipping.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {HF_VINE_ID} snapshot (~1.82 GB) ...")
    hf.snapshot_download(
        repo_id=HF_VINE_ID,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN,
        ignore_patterns=["*.git*", "setup_vine_complete.sh"],
    )
    print(f"  Saved to: {MODEL_DIR}")


def _clone_repo(name: str, url: str) -> None:
    dest = SRC_DIR / name
    if dest.exists():
        print(f"  Already cloned: {name}")
        return
    print(f"  Cloning {name} ...")
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth=1", url, str(dest)])
    print(f"  Saved to: {dest}")


def _register_vine_source() -> None:
    """
    Register model/ as a package root so `import vine` works.
    vine_model.py uses relative imports (from .vine_config import ...) so it
    must be imported as vine.vine_model, not as a top-level module.
    """
    package_root = str(MODEL_DIR.parent)  # model/
    init_file = MODEL_DIR / "__init__.py"
    init_file.touch(exist_ok=True)

    site_dir = Path(site.getsitepackages()[0])
    pth = site_dir / "vine_source.pth"
    if pth.exists() and pth.read_text().strip() == package_root:
        print("  vine source already registered.")
        return
    pth.write_text(package_root)
    print(f"  Registered vine package root → {pth}")


def _install_editable(name: str) -> None:
    dest = SRC_DIR / name
    print(f"  Installing {name} in editable mode ...")
    _run([sys.executable, "-m", "pip", "install", "-e", str(dest), "--quiet"])


def _download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already present: {dest.name}")
        return
    print(f"  Downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["wget", "-q", "-O", str(dest), url])
    print(f"  Saved: {dest.name}")


def main(include_detection: bool = False) -> None:
    hf = _require_hf_hub()
    if HF_TOKEN:
        print("HF_TOKEN found — using authenticated requests.")
    print()

    print("=== Model weights (video-fm/vine) ===")
    _download_vine_weights(hf)

    print("\n=== Vine source (model/vine/) → importable ===")
    _register_vine_source()

    print("\n=== Source repos ===")
    for name, url in REPOS.items():
        _clone_repo(name, url)
        _install_editable(name)

    if include_detection:
        print("\n=== Detection checkpoints (GroundingDINO + SAM2) ===")
        checkpoints = ROOT / "checkpoints"
        _download_file(HF_GDINO_URL, checkpoints / "groundingdino_swint_ogc.pth")
        _download_file(HF_SAM2_URL, checkpoints / "sam2.1_hiera_tiny.pt")

    MODEL_SENTINEL.touch()
    print(f"\nDone. Model in: {MODEL_DIR}")
    print(f"      Source in: {SRC_DIR}")
    print("""
To load the model:
    from vine.vine_model import VineModel
    from vine.vine_config import VineConfig
    cfg = VineConfig(use_hf_repo=False,
                     local_dir='model/vine',
                     local_filename='model.safetensors')
    model = VineModel.from_pretrained_vine('model/vine', config=cfg)
    model.eval()
""")


if __name__ == "__main__":
    detection = "--detection" in sys.argv
    try:
        main(include_detection=detection)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
