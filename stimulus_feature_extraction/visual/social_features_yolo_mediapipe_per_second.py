#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from social_features_yolo_mediapipe import (
    YOLO,
    TimeWindow,
    aggregate_rows_in_window,
    extract_frame_social_features,
    format_seconds,
    mp,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-second social features with YOLO + MediaPipe."
    )
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--output-dir",
        default="outputs/dynamic_social_4second_noalltime",
        help="Output root directory",
    )
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=5.0,
        help="Frames sampled per second before each one-second aggregation",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--close-threshold", type=float, default=0.15)
    parser.add_argument(
        "--duration-sec",
        type=int,
        help="Explicit movie duration in seconds; otherwise use detected duration",
    )
    return parser.parse_args()


def build_rows(
    total_duration_s: int,
    frame_rows: list[dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for second in range(max(0, total_duration_s)):
        window = TimeWindow(
            label=str(second),
            start_s=float(second),
            end_s=float(second + 1),
        )
        features = aggregate_rows_in_window(frame_rows, window)
        rows.append(
            {
                "second": second,
                "time_period": f"{format_seconds(second)}-{format_seconds(second + 1)}",
                "start_s": float(second),
                "end_s": float(second + 1),
                **{key: round(value, 6) for key, value in features.items()},
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed in the current environment")
    if mp is None:
        print("[warn] MediaPipe unavailable; using the OpenCV face fallback.")

    video_path = Path(args.video).expanduser().resolve()
    yolo_model: Any = YOLO(args.yolo_model)
    frame_rows, detected_duration_s = extract_frame_social_features(
        video_path=video_path,
        yolo_model=yolo_model,
        sample_fps=args.sample_fps,
        conf_thres=args.yolo_conf,
        close_thresh=args.close_threshold,
    )

    total_duration_s = (
        int(args.duration_sec)
        if args.duration_sec is not None
        else int(round(detected_duration_s))
    )
    rows = build_rows(total_duration_s, frame_rows)

    output_dir = Path(args.output_dir).expanduser().resolve() / video_path.stem
    output_path = output_dir / f"{video_path.stem}_social_features_per_second.csv"
    fieldnames = [
        "second",
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
    write_csv(output_path, rows, fieldnames)
    print(f"Done. Per-second social features: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
