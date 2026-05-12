"""
Download a subset of ESCA videos from YouTube using yt-dlp.
Only downloads videos that have masks AND a resolvable YouTube ID.

Supported sources:
  - liwei_youtube_videos (ytb_XXXXXXXXXXX)  →  8171 videos
  - activitynet          (v_XXXXXXXXXXX)    →    92 videos
  Skipped (not on YouTube): ego4d, Charades, NextQA, hash-named clips
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MASKS_DIR = DATASET_DIR / "mini_LLaVA_10000_masks"
VIDEOS_DIR = DATASET_DIR / "videos"
VIDEOS_SENTINEL = DATASET_DIR / ".videos_downloaded"

DEFAULT_MAX_VIDEOS = 500
YOUTUBE_BASE = "https://www.youtube.com/watch?v="

_SKIP_SOURCES = {"ego4d", "Charades", "NextQA", "youcook2"}


def _require_yt_dlp():
    try:
        import yt_dlp
        return yt_dlp
    except ImportError:
        print("yt-dlp is not installed. Run: pip install yt-dlp")
        sys.exit(1)


def _get_downloadable_videos() -> list[tuple[str, str]]:
    """
    Returns [(video_id, youtube_id), ...] for all mask-covered videos
    that have a resolvable YouTube ID.
    video_id  — the original ID used to name the mask file and frames dir
    youtube_id — the bare 11-char YouTube ID used for the download URL
    """
    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"Masks not extracted yet: {MASKS_DIR}")

    pairs = []
    for pkl in MASKS_DIR.rglob("*_mask.pkl"):
        parts = set(pkl.parts)

        if parts & _SKIP_SOURCES:
            continue

        video_id = pkl.stem.removesuffix("_mask")

        if "activitynet" in parts and video_id.startswith("v_"):
            pairs.append((video_id, video_id[2:]))          # v_XXXXXXXXXXX → XXXXXXXXXXX
        elif video_id.startswith("ytb_"):
            pairs.append((video_id, video_id[4:]))           # ytb_XXXXXXXXXXX → XXXXXXXXXXX
        # hash-named clips: no YouTube ID derivable — skip

    return sorted(pairs)


def main(max_videos: int = DEFAULT_MAX_VIDEOS) -> None:
    yt_dlp = _require_yt_dlp()

    print("Reading downloadable video IDs from mask subset...")
    all_pairs = _get_downloadable_videos()
    target_pairs = all_pairs[:max_videos]
    print(f"  {len(all_pairs)} YouTube-downloadable masked videos — targeting {len(target_pairs)}.")

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Stale sentinel: sentinel exists but videos directory is empty
    if VIDEOS_SENTINEL.exists() and not any(VIDEOS_DIR.iterdir()):
        print("  Stale sentinel found (no videos present) — re-downloading.")
        VIDEOS_SENTINEL.unlink()

    VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov")
    pending = [
        (vid, yt) for vid, yt in target_pairs
        if not any((VIDEOS_DIR / f"{vid}{ext}").exists() for ext in VIDEO_EXTS)
    ]
    print(f"  {len(target_pairs) - len(pending)} already downloaded, {len(pending)} to go.\n")

    if not pending:
        VIDEOS_SENTINEL.touch()
        print(f"All done. Videos in: {VIDEOS_DIR}")
        return

    failed = []
    # Use %(id)s naming then rename to video_id so frames can be matched back to masks
    ydl_opts = {
        "format": "best[height<=360]/worst",
        "outtmpl": str(VIDEOS_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, (video_id, yt_id) in enumerate(pending, 1):
            print(f"  [{i}/{len(pending)}] {video_id} ...", end=" ", flush=True)
            ydl.download([f"{YOUTUBE_BASE}{yt_id}"])

            # Rename yt_id.ext → video_id.ext
            renamed = False
            for f in VIDEOS_DIR.glob(f"{yt_id}.*"):
                f.rename(VIDEOS_DIR / f"{video_id}{f.suffix}")
                renamed = True
                break

            if renamed:
                print("ok")
            else:
                print("FAILED")
                failed.append(video_id)

    if failed:
        print(f"\n  {len(failed)} videos failed (unavailable/private/geo-blocked).")

    VIDEOS_SENTINEL.touch()
    print(f"\nDone. Videos in: {VIDEOS_DIR}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_VIDEOS
    try:
        main(max_videos=n)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
