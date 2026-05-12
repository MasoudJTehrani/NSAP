"""
Extract frames from downloaded ESCA video files at a target FPS.
Frames are saved as JPEG images under dataset/frames/{video_id}/{n:06d}.jpg
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
FRAMES_DIR = DATASET_DIR / "frames"
FRAMES_SENTINEL = DATASET_DIR / ".frames_extracted"

TARGET_FPS = 3
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".webm", ".mov"}


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        print("opencv-python is not installed. Run: pip install opencv-python")
        sys.exit(1)


def _find_videos() -> list[Path]:
    if not VIDEOS_DIR.exists():
        return []
    return [
        p for p in VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]


def _extract_video(cv2, video_path: Path, out_dir: Path, target_fps: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Warning: could not open {video_path.name}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    interval = max(1, round(src_fps / target_fps))

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            cv2.imwrite(str(out_dir / f"{saved:06d}.jpg"), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def main(target_fps: float = TARGET_FPS) -> None:
    cv2 = _require_cv2()

    videos = _find_videos()
    if not videos:
        print(f"No video files found in {VIDEOS_DIR}. Run download_videos.py first.")
        sys.exit(1)

    print(f"Found {len(videos)} videos. Extracting frames at {target_fps} FPS → {FRAMES_DIR}")
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    for i, vp in enumerate(videos, 1):
        video_id = vp.stem
        out_dir = FRAMES_DIR / video_id
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  [{i}/{len(videos)}] Already done: {video_id}")
            continue
        n = _extract_video(cv2, vp, out_dir, target_fps)
        print(f"  [{i}/{len(videos)}] {video_id}: {n} frames")

    FRAMES_SENTINEL.touch()
    print(f"\nDone. Frames saved to: {FRAMES_DIR}")


if __name__ == "__main__":
    fps = float(sys.argv[1]) if len(sys.argv) > 1 else TARGET_FPS
    try:
        main(target_fps=fps)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
