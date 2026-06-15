"""
Download ESCA videos from YouTube using yt-dlp.
Only downloads videos that have masks AND a resolvable YouTube ID.

Primary entry points
--------------------
ensure_videos_for_targets(targets)
    Download all clips that match the given attack targets (category /
    attribute / relation).  Already-present clips are skipped.  Called
    automatically by run_attack.py before each training/eval run.

cleanup_all_videos()
    Delete every video file in the videos directory.  Called automatically
    after training/eval unless --keep-videos is passed.

download_for_targets(targets, max_per_target)
    Same as ensure_videos_for_targets but with an explicit per-target cap.

main(max_videos)
    Legacy bulk download (used by first-time setup only).

Supported sources
-----------------
  liwei_youtube_videos  (ytb_XXXXXXXXXXX)  →  ~8171 videos
  activitynet           (v_XXXXXXXXXXX)    →    ~92 videos
  Skipped: ego4d, Charades, NextQA, hash-named clips (not on YouTube)

Reliability / rate-limit mitigation
-----------------------------------
YouTube rate-limits anonymous bulk downloads ("video unavailable, try again
later"). Downloads now retry, sleep between requests, and back off on a run of
failures. Tune via environment variables:

  NSAP_YTDLP_SLEEP            base seconds to sleep before each video (default 2;
                             actual sleep is randomised up to 3x this).
  NSAP_YTDLP_COOKIES         path to a cookies.txt (Netscape format) exported from
                             a logged-in browser — the single biggest win against
                             rate-limiting. Auto-detected if a cookies.txt sits in
                             the project root or dataset/ folder.
  NSAP_YTDLP_COOKIES_BROWSER browser name to pull cookies from directly instead of
                             a file (e.g. "firefox", "chrome"). Used only if no
                             cookie file is found.
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MASKS_DIR   = DATASET_DIR / "mini_LLaVA_10000_masks"
VIDEOS_DIR  = DATASET_DIR / "videos"
VIDEOS_SENTINEL = DATASET_DIR / ".videos_downloaded"

DEFAULT_MAX_VIDEOS = 500
YOUTUBE_BASE = "https://www.youtube.com/watch?v="
VIDEO_EXTS   = (".mp4", ".webm", ".mkv", ".avi", ".mov")

_SKIP_SOURCES = {"ego4d", "Charades", "NextQA", "youcook2"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_yt_dlp():
    try:
        import yt_dlp
        return yt_dlp
    except ImportError:
        print("yt-dlp is not installed. Run: pip install yt-dlp")
        sys.exit(1)


def _get_downloadable_videos() -> dict:
    """
    Returns {video_id: youtube_id} for all mask-covered clips with a
    resolvable YouTube ID.
    """
    if not MASKS_DIR.exists():
        raise FileNotFoundError(f"Masks not found: {MASKS_DIR}")

    result = {}
    for pkl in MASKS_DIR.rglob("*_mask.pkl"):
        parts = set(pkl.parts)
        if parts & _SKIP_SOURCES:
            continue
        video_id = pkl.stem.removesuffix("_mask")
        if "activitynet" in parts and video_id.startswith("v_"):
            result[video_id] = video_id[2:]
        elif video_id.startswith("ytb_"):
            result[video_id] = video_id[4:]
    return result


def _already_downloaded(video_id: str) -> bool:
    return any((VIDEOS_DIR / f"{video_id}{ext}").exists() for ext in VIDEO_EXTS)


def _datapoint_matches(datapoint: dict, targets: list) -> bool:
    """
    Return True if the datapoint satisfies ANY of the target dicts.
    Each target dict may contain keys: 'category', 'attribute', 'relation'.
    All specified keys must match (AND within a target, OR across targets).
    """
    consts      = [c.lower() for c in datapoint.get('consts',      [])]
    unary_kws   = [u.lower() for u in datapoint.get('unary_kws',   [])]
    binary_kws  = [b.lower() for b in datapoint.get('binary_kws',  [])]

    for target in targets:
        cat = target.get('category',  '').lower()
        att = target.get('attribute', '').lower()
        rel = target.get('relation',  '').lower()
        if cat and not any(cat in c for c in consts):
            continue
        if att and not any(att in u for u in unary_kws):
            continue
        if rel and not any(rel in b for b in binary_kws):
            continue
        return True
    return False


def _cookie_opts() -> dict:
    """
    Resolve yt-dlp cookie options from env / auto-detection.
    Priority: NSAP_YTDLP_COOKIES file > auto-detected cookies.txt > browser cookies.
    """
    cf = os.environ.get("NSAP_YTDLP_COOKIES", "").strip()
    if not cf:
        for cand in (ROOT / "cookies.txt", DATASET_DIR / "cookies.txt"):
            if cand.exists():
                cf = str(cand)
                break
    if cf and os.path.exists(cf):
        return {"cookiefile": cf}

    browser = os.environ.get("NSAP_YTDLP_COOKIES_BROWSER", "").strip()
    if browser:
        return {"cookiesfrombrowser": (browser.lower(),)}
    return {}


def _build_ydl_opts() -> dict:
    """yt-dlp options tuned to survive YouTube rate-limiting."""
    sleep = float(os.environ.get("NSAP_YTDLP_SLEEP", "2"))
    opts = {
        "format":       "best[height<=360]/worst",
        "outtmpl":      str(VIDEOS_DIR / "%(id)s.%(ext)s"),
        "quiet":        True,
        "no_warnings":  True,
        "ignoreerrors": True,
        # retry transient network / fragment / extractor failures
        "retries":            5,
        "fragment_retries":   5,
        "extractor_retries":  3,
        # throttle so YouTube is less likely to soft-block the session
        "sleep_interval":          sleep,                 # min sleep before a download
        "max_sleep_interval":      max(sleep, sleep * 3), # randomised up to 3x
        "sleep_interval_requests": 1,                     # sleep between metadata requests
    }
    opts.update(_cookie_opts())
    return opts


def _run_downloads(yt_dlp_mod, pending: list) -> list:
    """
    Download a list of (video_id, yt_id) pairs.
    Returns list of failed video_ids.

    Hardened against rate-limiting: retries + per-request sleep (see _build_ydl_opts),
    plus an adaptive cool-down when many downloads fail in a row.
    """
    if not pending:
        return []

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    failed = []

    ydl_opts = _build_ydl_opts()
    if "cookiefile" in ydl_opts:
        print(f"  Using cookies file: {ydl_opts['cookiefile']}")
    elif "cookiesfrombrowser" in ydl_opts:
        print(f"  Using cookies from browser: {ydl_opts['cookiesfrombrowser'][0]}")
    else:
        print("  No cookies (anonymous) — set NSAP_YTDLP_COOKIES to a cookies.txt "
              "to cut rate-limit failures.")
    print(f"  Throttle: sleep {ydl_opts['sleep_interval']}-{ydl_opts['max_sleep_interval']}s "
          f"between videos, retries={ydl_opts['retries']}.")

    COOLDOWN_AFTER = 8      # consecutive failures before a longer pause
    COOLDOWN_SECS  = 60     # how long to pause (let the rate-limit cool off)
    consecutive_fail = 0

    with yt_dlp_mod.YoutubeDL(ydl_opts) as ydl:
        for i, (video_id, yt_id) in enumerate(pending, 1):
            print(f"  [{i}/{len(pending)}] {video_id} ...", end=" ", flush=True)
            try:
                ydl.download([f"{YOUTUBE_BASE}{yt_id}"])
            except Exception:
                pass  # ignoreerrors handles most; this guards the rest

            renamed = False
            for f in VIDEOS_DIR.glob(f"{yt_id}.*"):
                f.rename(VIDEOS_DIR / f"{video_id}{f.suffix}")
                renamed = True
                break

            if renamed:
                print("ok")
                consecutive_fail = 0
            else:
                print("FAILED (unavailable / private / geo-blocked)")
                failed.append(video_id)
                consecutive_fail += 1
                if consecutive_fail and consecutive_fail % COOLDOWN_AFTER == 0:
                    print(f"  ⚠ {consecutive_fail} failures in a row — likely rate-limited; "
                          f"cooling down {COOLDOWN_SECS}s (Ctrl-C to stop).")
                    time.sleep(COOLDOWN_SECS)

    return failed


# ── public API ────────────────────────────────────────────────────────────────

def ensure_videos_for_targets(
    targets: list,
    annotation_file: str = None,
    max_per_target: int = 9999,
    split_stems: set = None,
) -> int:
    """
    Ensure all clips matching the given attack targets are downloaded.
    Already-present clips are skipped.  Returns the number of clips now
    available on disk for those targets.

    Args:
        targets:         list of dicts, e.g.
                           [{"category": "person", "attribute": "standing"}]
        annotation_file: path to JSONL (defaults to gpt_specs2_id_LLaVA_all.jsonl)
        max_per_target:  safety cap per individual target (default: no cap)
        split_stems:     if given, only download clips whose stem is in this set
                         (used to restrict downloads to train or test split only)

    Returns:
        number of matching video files now present on disk
    """
    yt_dlp = _require_yt_dlp()

    if annotation_file is None:
        annotation_file = str(DATASET_DIR / "gpt_specs2_id_LLaVA_all.jsonl")

    all_downloadable = _get_downloadable_videos()

    print("Scanning annotations for matching clips...")
    matched = []
    seen    = set()

    with open(annotation_file) as f:
        for line in f:
            if not line.strip():
                continue
            dp  = json.loads(line)
            vid = dp.get('video_id', '')
            prefix = next(
                (p for p in (f'ytb_{vid}', f'v_{vid}') if p in all_downloadable),
                None,
            )
            if prefix is None or prefix in seen:
                continue
            if _datapoint_matches(dp, targets):
                seen.add(prefix)
                matched.append(prefix)

    limit   = max_per_target * max(len(targets), 1)
    matched = matched[:limit]

    if split_stems is not None:
        matched = [vid for vid in matched if vid in split_stems]

    pending = [(vid, all_downloadable[vid]) for vid in matched if not _already_downloaded(vid)]
    already = len(matched) - len(pending)

    print(f"  {len(matched)} matching clips — {already} already on disk, {len(pending)} to download.")

    if pending:
        failed = _run_downloads(yt_dlp, pending)
        if failed:
            print(f"  {len(failed)} failed downloads (skipped).")

    on_disk = sum(1 for vid in matched if _already_downloaded(vid))
    print(f"  {on_disk} clips ready for training/eval.")
    VIDEOS_SENTINEL.touch()
    return on_disk


def cleanup_all_videos() -> int:
    """
    Delete all video files from the videos directory.
    Returns the number of files deleted.
    """
    if not VIDEOS_DIR.exists():
        return 0

    deleted = 0
    for f in VIDEOS_DIR.iterdir():
        if f.suffix in VIDEO_EXTS:
            f.unlink()
            deleted += 1

    print(f"  Removed {deleted} video file(s) from {VIDEOS_DIR}")
    return deleted


def download_for_targets(
    targets: list,
    max_per_target: int = 300,
    annotation_file: str = None,
) -> None:
    """Convenience wrapper around ensure_videos_for_targets with a per-target cap."""
    ensure_videos_for_targets(
        targets         = targets,
        annotation_file = annotation_file,
        max_per_target  = max_per_target,
    )


# ── legacy bulk download (first-time setup) ───────────────────────────────────

def main(max_videos: int = DEFAULT_MAX_VIDEOS) -> None:
    """
    Legacy: download the first max_videos from the mask-covered set.
    Used only by the first-time setup in NSAP.py — attacks use
    ensure_videos_for_targets() instead.
    """
    # In the new on-demand model the initial setup skips bulk downloads;
    # just touch the sentinel so the setup check passes.
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_SENTINEL.touch()
    print(f"  Video download deferred to per-attack on-demand mode.")
    print(f"  Run: python run_attack.py --mode <mode> --target <target>")
    print(f"  Videos will be downloaded automatically before each attack.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MAX_VIDEOS
    try:
        main(max_videos=n)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
