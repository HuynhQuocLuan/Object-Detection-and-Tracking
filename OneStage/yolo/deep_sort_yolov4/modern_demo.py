#!/usr/bin/env python3
"""CPU-friendly pedestrian detection demo for this subproject.

This replaces the unusable TensorFlow 1.x path with an OpenCV-only demo:
- MobileNet-SSD via OpenCV DNN for person detection
- simple centroid tracking for stable IDs across nearby frames
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2


PERSON_CLASS_ID = 15
CONFIDENCE_THRESHOLD = 0.5


@dataclass
class Track:
    track_id: int
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    misses: int = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_misses: int = 8) -> None:
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.next_id = 1
        self.tracks: dict[int, Track] = {}
        self.seen_ids: set[int] = set()

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, detections: list[tuple[int, int, int, int]]) -> dict[int, Track]:
        centers = [self._center(bbox) for bbox in detections]
        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for index, center in enumerate(centers):
                candidates.append((self._distance(track.center, center), track_id, index))

        for distance, track_id, index in sorted(candidates, key=lambda item: item[0]):
            if distance > self.max_distance:
                continue
            if track_id not in unmatched_tracks or index not in unmatched_detections:
                continue

            bbox = detections[index]
            self.tracks[track_id].center = centers[index]
            self.tracks[track_id].bbox = bbox
            self.tracks[track_id].misses = 0
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(index)

        for track_id in list(unmatched_tracks):
            self.tracks[track_id].misses += 1
            if self.tracks[track_id].misses > self.max_misses:
                del self.tracks[track_id]

        for index in unmatched_detections:
            bbox = detections[index]
            track = Track(
                track_id=self.next_id,
                center=centers[index],
                bbox=bbox,
            )
            self.tracks[track.track_id] = track
            self.seen_ids.add(track.track_id)
            self.next_id += 1

        return self.tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modern CPU-only pedestrian demo.")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to a local video file. Omit to use webcam 0.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow and only write the output video.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Write an annotated output video into the output folder.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 means no limit).",
    )
    return parser.parse_args()


def open_capture(project_dir: Path, input_path: str | None) -> tuple[cv2.VideoCapture, str]:
    if not input_path:
        capture = cv2.VideoCapture(0)
        return capture, "webcam:0"

    video_path = Path(input_path).expanduser()
    if not video_path.is_absolute():
        video_path = (project_dir / video_path).resolve()

    capture = cv2.VideoCapture(str(video_path))
    return capture, str(video_path)


def create_writer(
    project_dir: Path,
    capture: cv2.VideoCapture,
    enabled: bool,
) -> tuple[cv2.VideoWriter | None, Path | None]:
    if not enabled:
        return None, None

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 15.0

    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "modern_demo_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return writer, output_path


def load_detector(project_dir: Path) -> cv2.dnn_Net:
    models_dir = project_dir / "models"
    prototxt_path = models_dir / "mobilenet_ssd_deploy.prototxt"
    weights_path = models_dir / "mobilenet_iter_73000.caffemodel"
    if not prototxt_path.exists() or not weights_path.exists():
        missing = []
        if not prototxt_path.exists():
            missing.append(str(prototxt_path))
        if not weights_path.exists():
            missing.append(str(weights_path))
        raise FileNotFoundError("Missing detector model files:\n" + "\n".join(missing))

    return cv2.dnn.readNetFromCaffe(str(prototxt_path), str(weights_path))


def detect_people(
    net: cv2.dnn_Net,
    frame,
) -> list[tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5,
    )
    net.setInput(blob)
    detections = net.forward()
    boxes: list[tuple[int, int, int, int]] = []

    for index in range(detections.shape[2]):
        confidence = float(detections[0, 0, index, 2])
        class_id = int(detections[0, 0, index, 1])
        if class_id != PERSON_CLASS_ID or confidence < CONFIDENCE_THRESHOLD:
            continue

        x1 = int(detections[0, 0, index, 3] * width)
        y1 = int(detections[0, 0, index, 4] * height)
        x2 = int(detections[0, 0, index, 5] * width)
        y2 = int(detections[0, 0, index, 6] * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        if box_width == 0 or box_height == 0:
            continue
        boxes.append((x1, y1, box_width, box_height))

    return boxes


def annotate_frame(frame, tracks: dict[int, Track], total_seen: int) -> None:
    current_count = 0
    for track in tracks.values():
        if track.misses:
            continue
        current_count += 1
        x, y, w, h = track.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 60), 2)
        cv2.putText(
            frame,
            f"ID {track.track_id}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 180, 255),
            2,
        )

    cv2.putText(
        frame,
        f"Current: {current_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Total IDs: {total_seen}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def main() -> int:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent

    print("Starting modern DNN demo...", flush=True)
    print("Opening input source...", flush=True)
    capture, source_label = open_capture(project_dir, args.input)
    if not capture.isOpened():
        print(f"Failed to open input source: {source_label}", flush=True)
        return 1

    writer, output_path = create_writer(project_dir, capture, args.save_output)

    try:
        net = load_detector(project_dir)
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        capture.release()
        return 1

    tracker = CentroidTracker()

    frame_index = 0
    print(f"Running modern CPU demo on: {source_label}", flush=True)
    print("Detector: MobileNet-SSD (OpenCV DNN)", flush=True)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        detections = detect_people(net, frame)
        tracks = tracker.update(detections)
        annotate_frame(frame, tracks, len(tracker.seen_ids))

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Modern CPU Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_index += 1
        if args.max_frames > 0 and frame_index >= args.max_frames:
            break

    capture.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"Frames processed: {frame_index}", flush=True)
    print(f"Tracked IDs seen: {len(tracker.seen_ids)}", flush=True)
    if output_path is not None:
        print(f"Annotated output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
