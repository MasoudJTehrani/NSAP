"""
Download the VINE model (video-fm/vine) — the public release of SGCLIP/LASER.

Downloads:
  model/vine/    — HuggingFace snapshot (weights + vine source files, ~1.82 GB)
  src/LASER/     — LASER training code (PredicateModel, Scallop pipeline)

The vine source files (vine_model.py, vine_pipeline.py, vine_config.py) ship
directly inside the HF repo, so no separate vine_hf package clone is needed.
They are registered via a .pth file so they are importable as top-level modules.

SAM2 is required (anchor sub-region selection during training) and is fetched
separately by NSAP.py via download_sam2(). GroundingDINO is optional — the sim
dataset ships ground-truth masks, so the detection frontend is not needed for the
attack itself; pass include_detection=True to fetch it anyway.
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

# --- Frozen sim dataset (SceneGraphBench snapshot) on the HuggingFace Hub ---
HF_DATA_ID        = "MasoudJTehrani/NSAP-sim-data"
# Pinned to the dataset commit sha for EXACT reproducibility (a mutable branch like
# 'main' would drift if the repo is ever changed). Override with $NSAP_DATA_REVISION.
HF_DATA_REVISION  = os.environ.get("NSAP_DATA_REVISION", "cfc526c9d27c9ba16f8df097da2490abe0acce2b")
SIM_DATA_DIR      = ROOT / "dataset" / "sim"
# The EB-Nav deployment scenes (kitchen_ebnav) live in the SAME dataset repo under a
# kitchen_ebnav/ subfolder, fetched as a sibling of sim (used by the category
# chameleons and the MIM). See download_dataset().
EBNAV_DATA_DIR    = ROOT / "dataset" / "kitchen_ebnav"
SIM_DATA_SENTINEL = DATASET_DIR / ".sim_data_downloaded"


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

    # site.getsitepackages() is missing in some virtualenvs — fall back to sysconfig.
    try:
        site_dir = Path(site.getsitepackages()[0])
    except (AttributeError, IndexError):
        import sysconfig
        site_dir = Path(sysconfig.get_path("purelib"))
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
    # stdlib download (portable — no wget/curl dependency); write to a .part temp
    # then atomically rename so an interrupted download can't leave a corrupt file
    # that the idempotent existence check above would later skip.
    import urllib.request
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"  Download failed: {e}")
        sys.exit(1)
    print(f"  Saved: {dest.name}")


SAM2_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_SENTINEL   = DATASET_DIR / ".sam2_downloaded"


def download_sam2() -> None:
    if SAM2_SENTINEL.exists():
        print("  SAM2 checkpoint already present — skipping.")
        return
    _download_file(HF_SAM2_URL, SAM2_CHECKPOINT)
    SAM2_SENTINEL.touch()


def download_dataset(revision: str | None = None) -> None:
    """Download the frozen SceneGraphBench sim dataset snapshot from the HF Hub.

    Pin `revision` (a commit sha, via $NSAP_DATA_REVISION) for exact reproducibility;
    defaults to 'main'. Idempotent (sentinel-guarded). The dataset repo is public, so
    no token is required.
    """
    hf = _require_hf_hub()
    if SIM_DATA_SENTINEL.exists():
        print("  Sim dataset already present, skipping.")
        return
    rev = revision or HF_DATA_REVISION
    # The repo holds the merged sim dataset at its ROOT (manifest.json + scenarios/)
    # plus a kitchen_ebnav/ subfolder for the EB-Nav deployment scenes. Lay them out
    # as siblings locally: sim -> dataset/sim, kitchen_ebnav -> dataset/kitchen_ebnav.
    SIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading dataset {HF_DATA_ID}@{rev} -> {SIM_DATA_DIR} ...")
    hf.snapshot_download(
        repo_id=HF_DATA_ID,
        repo_type="dataset",
        revision=rev,
        local_dir=str(SIM_DATA_DIR),
        token=HF_TOKEN,
        ignore_patterns=["kitchen_ebnav/*"],
    )
    print(f"  Downloading kitchen_ebnav -> {EBNAV_DATA_DIR} ...")
    hf.snapshot_download(
        repo_id=HF_DATA_ID,
        repo_type="dataset",
        revision=rev,
        local_dir=str(DATASET_DIR),
        token=HF_TOKEN,
        allow_patterns=["kitchen_ebnav/*"],
    )
    SIM_DATA_SENTINEL.touch()
    print(f"  Saved to: {SIM_DATA_DIR}  and  {EBNAV_DATA_DIR}")


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
        print("\n=== Detection checkpoints (GroundingDINO) ===")
        _download_file(HF_GDINO_URL, ROOT / "checkpoints" / "groundingdino_swint_ogc.pth")

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
