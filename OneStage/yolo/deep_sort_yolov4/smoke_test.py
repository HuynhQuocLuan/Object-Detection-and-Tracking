#!/usr/bin/env python3
"""Lightweight runtime check for the YOLOv4 Deep SORT subproject.

This avoids the legacy TensorFlow stack and only validates the local
environment, project layout, and optional video decoding path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a lightweight smoke test for YOLOv4 Deep SORT."
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Optional path to a local video file to validate decoding.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Maximum number of frames to decode when --input is provided.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "output"
    preview_path = output_dir / "smoke_preview.jpg"

    required_paths = [
        project_dir / "deep_sort",
        project_dir / "model_data" / "market1501.pb",
        output_dir,
    ]
    optional_weights = project_dir / "model_data" / "yolo4_weight.h5"

    print("YOLOv4 Deep SORT smoke test")
    print(f"python: {sys.executable}")
    print(f"opencv: {cv2.__version__}")
    print(f"project: {project_dir}")

    missing_required = [path for path in required_paths if not path.exists()]
    if missing_required:
        print("missing required project files:")
        for path in missing_required:
            print(f"  - {path}")
        return 1

    if optional_weights.exists():
        print(f"weights: found {optional_weights.name}")
    else:
        print(f"weights: missing {optional_weights.name} (full detection will not run)")

    if not args.input:
        print("environment check passed")
        print("tip: rerun with --input /absolute/path/to/video.mp4 to test video decoding")
        return 0

    video_path = Path(args.input).expanduser()
    if not video_path.is_absolute():
        video_path = (project_dir / video_path).resolve()

    if not video_path.exists():
        print(f"input video not found: {video_path}")
        return 1

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"failed to open video: {video_path}")
        return 1

    frame_count = 0
    first_frame = None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    while frame_count < max(args.max_frames, 1):
        ok, frame = capture.read()
        if not ok:
            break
        if first_frame is None:
            first_frame = frame.copy()
        frame_count += 1

    capture.release()

    if first_frame is None:
        print(f"no frames decoded from: {video_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), first_frame)

    print(f"video: {video_path}")
    print(f"frames decoded: {frame_count}")
    print(f"frame size: {width}x{height}")
    print(f"source fps: {fps:.2f}")
    print(f"preview written: {preview_path}")
    print("video smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
