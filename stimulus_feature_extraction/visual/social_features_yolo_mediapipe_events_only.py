#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from social_features_yolo_mediapipe import (
    YOLO,
    aggregate_rows_in_window,
    extract_frame_social_features,
    mp,
    read_event_windows,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract event-only social features with YOLO + MediaPipe.")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--segments-csv", required=True, help="Path to event segmentation CSV")
    parser.add_argument("--output-dir", default="outputs/dynamic_social", help="Output root directory")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path/name for person detection")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frame sampling FPS")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--close-threshold", type=float, default=0.15, help="Normalized person distance threshold for close pairs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed in current env. Run: uv pip install ultralytics")
    if mp is None:
        print("[warn] mediapipe is not installed. Falling back to OpenCV face detector for face-related features.")

    video_path = Path(args.video).expanduser().resolve()
    segments_csv = Path(args.segments_csv).expanduser().resolve()
    event_windows = read_event_windows(segments_csv)

    yolo_model: Any = YOLO(args.yolo_model)
    frame_rows, _duration_s = extract_frame_social_features(
        video_path=video_path,
        yolo_model=yolo_model,
        sample_fps=args.sample_fps,
        conf_thres=args.yolo_conf,
        close_thresh=args.close_threshold,
    )

    video_stem = video_path.stem
    out_dir = Path(args.output_dir).expanduser().resolve() / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "window_id",
        "time_period",
        "start_s",
        "end_s",
        "people_count",
        "people_count_max",
        "people_count_p90",
        "people_presence_ratio",
        "person_area_mean",
        "people_close_count",
        "face_count",
        "face_count_max",
        "face_count_p90",
        "face_presence_ratio",
        "face_area_mean",
        "interpersonal_distance_mean",
        "mutual_gaze_score",
        "facial_expression_intensity",
        "gesture_intensity",
        "happy",
        "sad",
        "angry",
        "calm",
        "fearful",
    ]

    event_rows: list[dict[str, Any]] = []
    for w in event_windows:
        agg = aggregate_rows_in_window(frame_rows, w)
        event_rows.append(
            {
                "window_id": w.label,
                "time_period": w.time_period,
                "start_s": round(w.start_s, 3),
                "end_s": round(w.end_s, 3),
                **{k: round(v, 6) for k, v in agg.items()},
            }
        )

    event_path = out_dir / f"{video_stem}_social_features_events.csv"
    write_csv(event_path, event_rows, fieldnames)
    print(f"Done. Event social features: {event_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
