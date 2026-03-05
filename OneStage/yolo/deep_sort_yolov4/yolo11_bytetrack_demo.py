#!/usr/bin/env python3
"""YOLO11 + ByteTrack demo for webcam/video object detection and tracking."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import cv2
from ultralytics import YOLO


COCO80_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO11 + ByteTrack demo.")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to a local video file. Omit to use webcam 0.",
    )
    parser.add_argument(
        "-c",
        "--classes",
        default="person",
        help="Comma-separated class names, e.g. 'person,car,bus'.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--line-ratio",
        type=float,
        default=0.5,
        help="Vertical count line position as frame-width ratio (0.0-1.0).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Model path. Defaults to local yolo11n.pt in this folder.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow and only write the output video.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Write annotated output video into the output folder.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 means no limit).",
    )
    return parser.parse_args()


def parse_class_ids(value: str) -> list[int]:
    wanted = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not wanted:
        raise ValueError("No classes provided.")

    ids: list[int] = []
    for name in wanted:
        if name not in COCO80_NAMES:
            raise ValueError(f"Unsupported class '{name}'.")
        ids.append(COCO80_NAMES.index(name))
    return ids


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
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"yolo11_bytetrack_{stamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return writer, output_path


def draw_boxes(frame, result, track_ids_seen: set[int]) -> tuple[int, dict[int, int]]:
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return 0, {}

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
    cls_ids = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(xyxy)

    active = 0
    current_centers_x: dict[int, int] = {}
    for idx, coords in enumerate(xyxy):
        x1, y1, x2, y2 = coords.tolist()
        conf = float(confs[idx]) if len(confs) > idx else 0.0
        class_id = int(cls_ids[idx]) if len(cls_ids) > idx else -1
        class_name = COCO80_NAMES[class_id] if 0 <= class_id < len(COCO80_NAMES) else str(class_id)
        track_id = int(track_ids[idx]) if len(track_ids) > idx else -1
        if track_id >= 0:
            track_ids_seen.add(track_id)
            current_centers_x[track_id] = (x1 + x2) // 2
        active += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
        cv2.putText(
            frame,
            f"{class_name} ID:{track_id} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 180, 255),
            2,
        )

    return active, current_centers_x


def main() -> int:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent

    try:
        class_ids = parse_class_ids(args.classes)
    except ValueError as exc:
        print(f"Class parse error: {exc}", flush=True)
        return 1

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = (project_dir / model_path).resolve()
    if not model_path.exists():
        print(f"Model file not found: {model_path}", flush=True)
        return 1

    print("Starting YOLO11 + ByteTrack demo...", flush=True)
    print(f"Model: {model_path}", flush=True)
    print(f"Classes: {args.classes}", flush=True)

    capture, source_label = open_capture(project_dir, args.input)
    if not capture.isOpened():
        print(f"Failed to open input source: {source_label}", flush=True)
        return 1

    writer, output_path = create_writer(project_dir, capture, args.save_output)
    model = YOLO(str(model_path))
    seen_track_ids: set[int] = set()
    previous_centers_x: dict[int, int] = {}
    counted_ids: set[int] = set()
    in_count = 0
    out_count = 0
    line_x: int | None = None
    frame_index = 0

    print(f"Input source: {source_label}", flush=True)
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            classes=class_ids,
            verbose=False,
        )
        result = results[0]
        active_count, current_centers_x = draw_boxes(frame, result, seen_track_ids)
        if line_x is None:
            line_ratio = min(1.0, max(0.0, args.line_ratio))
            line_x = int(frame.shape[1] * line_ratio)

        for track_id, current_x in current_centers_x.items():
            if track_id in counted_ids:
                continue
            previous_x = previous_centers_x.get(track_id)
            if previous_x is None:
                continue
            # IN: left-to-right crossing. OUT: right-to-left crossing.
            if previous_x < line_x <= current_x:
                in_count += 1
                counted_ids.add(track_id)
            elif previous_x > line_x >= current_x:
                out_count += 1
                counted_ids.add(track_id)

        previous_centers_x = current_centers_x

        cv2.putText(
            frame,
            f"Active tracks: {active_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Total IDs: {len(seen_track_ids)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"IN: {in_count} OUT: {out_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.line(
            frame,
            (line_x, 0),
            (line_x, frame.shape[0]),
            (255, 255, 0),
            2,
        )

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("YOLO11 + ByteTrack", frame)
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
    print(f"Tracked IDs seen: {len(seen_track_ids)}", flush=True)
    print(f"Line crossing counts -> IN: {in_count}, OUT: {out_count}", flush=True)
    if output_path is not None:
        print(f"Annotated output: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
