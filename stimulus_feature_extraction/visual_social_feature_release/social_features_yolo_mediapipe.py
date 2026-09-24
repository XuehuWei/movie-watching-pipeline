#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


@dataclass
class TimeWindow:
    label: str
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def time_period(self) -> str:
        return f"{format_seconds(self.start_s)}-{format_seconds(self.end_s)}"


def format_seconds(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def parse_mmss(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        mm, ss = parts
        return float(int(mm) * 60 + int(ss))
    if len(parts) == 3:
        hh, mm, ss = parts
        return float(int(hh) * 3600 + int(mm) * 60 + int(ss))
    raise ValueError(f"Unsupported time format: {value}")


def read_event_windows(segmentation_csv: Path) -> list[TimeWindow]:
    windows: list[TimeWindow] = []
    with segmentation_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            label = (row.get("Event") or f"Event {idx}").strip()
            start_txt = (row.get("Start (mm:ss)") or "").strip()
            end_txt = (row.get("End (mm:ss)") or "").strip()
            if not start_txt or not end_txt:
                continue
            start_s = parse_mmss(start_txt)
            end_s = parse_mmss(end_txt)
            if end_s <= start_s:
                continue
            windows.append(TimeWindow(label=label, start_s=start_s, end_s=end_s))
    if not windows:
        raise RuntimeError(f"No valid event windows in: {segmentation_csv}")
    return windows


def build_sliding_windows(duration_s: float, win_len_s: float, step_s: float) -> list[TimeWindow]:
    windows: list[TimeWindow] = []
    idx = 0
    start = 0.0
    while start + win_len_s <= duration_s + 1e-9:
        end = start + win_len_s
        windows.append(TimeWindow(label=f"SW_{idx:05d}", start_s=start, end_s=end))
        start += step_s
        idx += 1
    return windows


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _landmark_xy(lm: Any, w: int, h: int, idx: int) -> tuple[float, float]:
    p = lm[idx]
    return p.x * w, p.y * h


def _dist2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def pairwise_dist_stats(centers: list[tuple[float, float]], diag: float, close_thresh: float) -> tuple[float, int]:
    if len(centers) < 2:
        return 0.0, 0
    dists: list[float] = []
    close_count = 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = _dist2d(centers[i], centers[j]) / max(1e-8, diag)
            dists.append(d)
            if d <= close_thresh:
                close_count += 1
    return (float(np.mean(dists)) if dists else 0.0), close_count


def match_centers_and_motion(
    prev_centers: list[tuple[float, float]],
    curr_centers: list[tuple[float, float]],
    diag: float,
) -> float:
    if not prev_centers or not curr_centers:
        return 0.0
    used_prev: set[int] = set()
    disp: list[float] = []
    for c in curr_centers:
        best_i = None
        best_d = None
        for i, p in enumerate(prev_centers):
            if i in used_prev:
                continue
            d = _dist2d(c, p)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        if best_i is not None and best_d is not None:
            used_prev.add(best_i)
            disp.append(best_d / max(1e-8, diag))
    return float(np.mean(disp)) if disp else 0.0


def _face_stats_from_landmarks(face_landmarks: Any, w: int, h: int) -> dict[str, float]:
    xs = [p.x * w for p in face_landmarks.landmark]
    ys = [p.y * h for p in face_landmarks.landmark]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    face_w = max(1.0, max_x - min_x)
    face_h = max(1.0, max_y - min_y)
    face_area_norm = (face_w * face_h) / max(1.0, float(w * h))

    left_eye_outer = _landmark_xy(face_landmarks.landmark, w, h, 33)
    right_eye_outer = _landmark_xy(face_landmarks.landmark, w, h, 263)
    nose_tip = _landmark_xy(face_landmarks.landmark, w, h, 1)
    eye_dist = max(1.0, _dist2d(left_eye_outer, right_eye_outer))
    eye_mid_x = (left_eye_outer[0] + right_eye_outer[0]) / 2.0
    yaw_proxy = abs((nose_tip[0] - eye_mid_x) / (eye_dist / 2.0))
    frontal = clamp01(1.0 - min(1.0, yaw_proxy))

    mouth_left = _landmark_xy(face_landmarks.landmark, w, h, 61)
    mouth_right = _landmark_xy(face_landmarks.landmark, w, h, 291)
    lip_top = _landmark_xy(face_landmarks.landmark, w, h, 13)
    lip_bottom = _landmark_xy(face_landmarks.landmark, w, h, 14)
    brow_left = _landmark_xy(face_landmarks.landmark, w, h, 70)
    brow_right = _landmark_xy(face_landmarks.landmark, w, h, 300)
    l_eye_up = _landmark_xy(face_landmarks.landmark, w, h, 159)
    l_eye_dn = _landmark_xy(face_landmarks.landmark, w, h, 145)
    r_eye_up = _landmark_xy(face_landmarks.landmark, w, h, 386)
    r_eye_dn = _landmark_xy(face_landmarks.landmark, w, h, 374)

    mouth_width = max(1.0, _dist2d(mouth_left, mouth_right))
    mouth_open = _dist2d(lip_top, lip_bottom) / mouth_width
    eye_open = 0.5 * (_dist2d(l_eye_up, l_eye_dn) + _dist2d(r_eye_up, r_eye_dn)) / eye_dist
    brow_dist = _dist2d(brow_left, brow_right) / eye_dist
    brow_furrow = clamp01((0.45 - brow_dist) / 0.25)
    smile_proxy = clamp01((mouth_width / max(1.0, face_w) - 0.33) / 0.18)
    expr_intensity = clamp01(0.45 * mouth_open + 0.35 * brow_furrow + 0.20 * abs(eye_open - 0.08) * 6.0)

    happy_raw = clamp01(0.65 * smile_proxy + 0.35 * max(0.0, 0.10 - mouth_open) * 4.0)
    sad_raw = clamp01(0.45 * (1.0 - smile_proxy) + 0.35 * max(0.0, 0.07 - eye_open) * 6.0 + 0.20 * mouth_open)
    angry_raw = clamp01(0.70 * brow_furrow + 0.30 * expr_intensity)
    fearful_raw = clamp01(0.55 * max(0.0, eye_open - 0.08) * 8.0 + 0.45 * mouth_open)
    calm_raw = clamp01(max(0.0, 1.0 - expr_intensity))

    vec = np.array([sad_raw, happy_raw, angry_raw, calm_raw, fearful_raw], dtype=np.float64)
    vec = vec / max(1e-8, float(np.sum(vec)))

    return {
        "face_area_norm": float(face_area_norm),
        "frontal_score": float(frontal),
        "expr_intensity": float(expr_intensity),
        "sad": float(vec[0]),
        "happy": float(vec[1]),
        "angry": float(vec[2]),
        "calm": float(vec[3]),
        "fearful": float(vec[4]),
        "face_center_x": float((min_x + max_x) * 0.5),
        "face_center_y": float((min_y + max_y) * 0.5),
    }


def _face_stats_from_bbox(gray: np.ndarray, x: int, y: int, fw: int, fh: int, w: int, h: int) -> dict[str, float]:
    """Fallback face stats when MediaPipe FaceMesh is unavailable.

    We estimate expression intensity from ROI appearance dynamics proxies so the
    feature is not hard-coded to zero in OpenCV-only mode.
    """
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(w, int(x + fw))
    y1 = min(h, int(y + fh))
    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        expr_intensity = 0.0
        smile_proxy = 0.0
        brow_proxy = 0.0
    else:
        # Texture/edge strength increases with expressive facial changes.
        lap_var = float(cv2.Laplacian(roi, cv2.CV_32F).var())
        edge_density = float(np.mean(cv2.Canny(roi, 60, 120) > 0))

        hh, ww = roi.shape[:2]
        upper = roi[: max(1, hh // 2), :]
        lower = roi[max(0, hh // 2) :, :]
        upper_std = float(np.std(upper)) if upper.size else 0.0
        lower_std = float(np.std(lower)) if lower.size else 0.0

        # Mouth region activity proxy: stronger lower-half variability.
        smile_proxy = clamp01((lower_std - 14.0) / 22.0)
        brow_proxy = clamp01((upper_std - 12.0) / 20.0)
        dyn_proxy = clamp01((lap_var - 40.0) / 180.0)
        edge_proxy = clamp01((edge_density - 0.06) / 0.20)
        expr_intensity = clamp01(0.40 * dyn_proxy + 0.35 * edge_proxy + 0.25 * max(smile_proxy, brow_proxy))

    # Coarse emotion proxies in fallback mode.
    happy_raw = clamp01(0.70 * smile_proxy + 0.30 * expr_intensity)
    angry_raw = clamp01(0.55 * brow_proxy + 0.45 * expr_intensity)
    sad_raw = clamp01(0.45 * (1.0 - smile_proxy) + 0.55 * max(0.0, 0.35 - expr_intensity))
    fearful_raw = clamp01(0.40 * expr_intensity + 0.30 * brow_proxy + 0.30 * (1.0 - smile_proxy))
    calm_raw = clamp01(1.0 - expr_intensity)
    vec = np.array([sad_raw, happy_raw, angry_raw, calm_raw, fearful_raw], dtype=np.float64)
    vec = vec / max(1e-8, float(np.sum(vec)))

    cx = x + fw / 2.0
    cy = y + fh / 2.0
    return {
        "face_area_norm": float((fw * fh) / max(1.0, float(w * h))),
        "frontal_score": 0.5,
        "expr_intensity": float(expr_intensity),
        "sad": float(vec[0]),
        "happy": float(vec[1]),
        "angry": float(vec[2]),
        "calm": float(vec[3]),
        "fearful": float(vec[4]),
        "face_center_x": float(cx),
        "face_center_y": float(cy),
    }


def extract_frame_social_features(
    video_path: Path,
    yolo_model: Any,
    sample_fps: float,
    conf_thres: float,
    close_thresh: float,
) -> tuple[list[dict[str, float]], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / native_fps if total_frames > 0 else 0.0
    step = max(1, int(round(native_fps / max(0.1, sample_fps))))

    # Newer mediapipe wheels (e.g. py3.13) may expose only `mediapipe.tasks`
    # and not `mp.solutions`. In that case we fallback to OpenCV face detector.
    mp_face_mesh = None
    if mp is not None and hasattr(mp, "solutions"):
        try:
            mp_face_mesh = mp.solutions.face_mesh.FaceMesh(  # type: ignore[union-attr]
                static_image_mode=True,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except Exception as exc:
            print(f"[warn] MediaPipe FaceMesh init failed, fallback to OpenCV faces: {exc}")
            mp_face_mesh = None
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    idx = 0
    prev_person_centers: list[tuple[float, float]] = []
    rows: list[dict[str, float]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        h, w = frame.shape[:2]
        diag = math.sqrt(float(h * h + w * w))
        t_sec = idx / native_fps

        pred = yolo_model.predict(frame, verbose=False, conf=conf_thres)
        person_boxes = []
        if pred and len(pred) > 0:
            boxes = pred[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
                for b, c in zip(xyxy, cls):
                    if int(c) == 0:
                        x1, y1, x2, y2 = map(float, b.tolist())
                        person_boxes.append((x1, y1, x2, y2))

        person_centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in person_boxes]
        person_area_mean = (
            float(
                np.mean(
                    [
                        max(0.0, (b[2] - b[0]) * (b[3] - b[1])) / max(1.0, float(w * h))
                        for b in person_boxes
                    ]
                )
            )
            if person_boxes
            else 0.0
        )
        interpersonal_distance_mean, people_close_count = pairwise_dist_stats(person_centers, diag, close_thresh)
        gesture_intensity = match_centers_and_motion(prev_person_centers, person_centers, diag)
        prev_person_centers = person_centers

        face_stats: list[dict[str, float]] = []
        if mp_face_mesh is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh_out = mp_face_mesh.process(rgb)
            if mesh_out.multi_face_landmarks:
                for lm in mesh_out.multi_face_landmarks:
                    face_stats.append(_face_stats_from_landmarks(lm, w, h))
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            for (x, y, fw, fh) in faces:
                face_stats.append(_face_stats_from_bbox(gray, int(x), int(y), int(fw), int(fh), w, h))

        face_count = float(len(face_stats))
        face_area_mean = float(np.mean([x["face_area_norm"] for x in face_stats])) if face_stats else 0.0
        facial_expression_intensity = float(np.mean([x["expr_intensity"] for x in face_stats])) if face_stats else 0.0
        sad = float(np.mean([x["sad"] for x in face_stats])) if face_stats else 0.0
        happy = float(np.mean([x["happy"] for x in face_stats])) if face_stats else 0.0
        angry = float(np.mean([x["angry"] for x in face_stats])) if face_stats else 0.0
        calm = float(np.mean([x["calm"] for x in face_stats])) if face_stats else 1.0
        fearful = float(np.mean([x["fearful"] for x in face_stats])) if face_stats else 0.0

        if len(face_stats) >= 2:
            fcenters = [(x["face_center_x"], x["face_center_y"]) for x in face_stats]
            fdist_mean, _ = pairwise_dist_stats(fcenters, diag, close_thresh=0.20)
            frontal = float(np.mean([x["frontal_score"] for x in face_stats]))
            mutual_gaze_score = clamp01(frontal * max(0.0, 1.0 - (fdist_mean / 0.35)))
        elif len(face_stats) == 1:
            mutual_gaze_score = float(face_stats[0]["frontal_score"] * 0.3)
        else:
            mutual_gaze_score = 0.0

        emo_vec = np.array([sad, happy, angry, calm, fearful], dtype=np.float64)
        emo_vec = emo_vec / max(1e-8, float(np.sum(emo_vec)))
        sad, happy, angry, calm, fearful = map(float, emo_vec.tolist())

        rows.append(
            {
                "t_sec": float(t_sec),
                "people_count": float(len(person_boxes)),
                "person_area_mean": person_area_mean,
                "people_close_count": float(people_close_count),
                "face_count": face_count,
                "face_area_mean": face_area_mean,
                "interpersonal_distance_mean": interpersonal_distance_mean,
                "mutual_gaze_score": mutual_gaze_score,
                "facial_expression_intensity": facial_expression_intensity,
                "gesture_intensity": gesture_intensity,
                "sad": sad,
                "happy": happy,
                "angry": angry,
                "calm": calm,
                "fearful": fearful,
            }
        )
        idx += 1

    cap.release()
    if mp_face_mesh is not None:
        mp_face_mesh.close()
    return rows, float(duration_s)


def aggregate_rows_in_window(frame_rows: list[dict[str, float]], window: TimeWindow) -> dict[str, float]:
    sel = [r for r in frame_rows if window.start_s <= r["t_sec"] < window.end_s]
    if not sel:
        return {
            "people_count": 0.0,
            "person_area_mean": 0.0,
            "people_close_count": 0.0,
            "face_count": 0.0,
            "face_area_mean": 0.0,
            "interpersonal_distance_mean": 0.0,
            "mutual_gaze_score": 0.0,
            "facial_expression_intensity": 0.0,
            "gesture_intensity": 0.0,
            "sad": 0.0,
            "happy": 0.0,
            "angry": 0.0,
            "calm": 1.0,
            "fearful": 0.0,
        }

    keys = [
        "people_count",
        "person_area_mean",
        "people_close_count",
        "face_count",
        "face_area_mean",
        "interpersonal_distance_mean",
        "mutual_gaze_score",
        "facial_expression_intensity",
        "gesture_intensity",
        "sad",
        "happy",
        "angry",
        "calm",
        "fearful",
    ]
    out = {k: float(np.mean([r[k] for r in sel])) for k in keys}

    people_vals = np.array([r["people_count"] for r in sel], dtype=np.float64)
    face_vals = np.array([r["face_count"] for r in sel], dtype=np.float64)
    out["people_count_max"] = float(np.max(people_vals)) if people_vals.size else 0.0
    out["people_count_p90"] = float(np.percentile(people_vals, 90)) if people_vals.size else 0.0
    out["people_presence_ratio"] = float(np.mean(people_vals > 0.0)) if people_vals.size else 0.0
    out["face_count_max"] = float(np.max(face_vals)) if face_vals.size else 0.0
    out["face_count_p90"] = float(np.percentile(face_vals, 90)) if face_vals.size else 0.0
    out["face_presence_ratio"] = float(np.mean(face_vals > 0.0)) if face_vals.size else 0.0

    emo = np.array([out["sad"], out["happy"], out["angry"], out["calm"], out["fearful"]], dtype=np.float64)
    emo = emo / max(1e-8, float(np.sum(emo)))
    out["sad"], out["happy"], out["angry"], out["calm"], out["fearful"] = map(float, emo.tolist())
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract social features with YOLO + MediaPipe (sliding/event windows).")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--segments-csv", required=True, help="Path to event segmentation CSV")
    parser.add_argument("--output-dir", default="outputs/dynamic_social", help="Output root directory")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path/name for person detection")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frame sampling FPS")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--close-threshold", type=float, default=0.15, help="Normalized person distance threshold for close pairs")
    parser.add_argument("--sliding-window-sec", type=float, default=40.0, help="Sliding window length in seconds")
    parser.add_argument("--sliding-step-sec", type=float, default=1.0, help="Sliding window step in seconds")
    parser.add_argument("--duration-sec", type=float, help="Optional total video duration in seconds for full sliding-window coverage")
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

    yolo_model = YOLO(args.yolo_model)
    frame_rows, duration_s = extract_frame_social_features(
        video_path=video_path,
        yolo_model=yolo_model,
        sample_fps=args.sample_fps,
        conf_thres=args.yolo_conf,
        close_thresh=args.close_threshold,
    )

    total_duration_s = float(args.duration_sec) if args.duration_sec else duration_s
    sliding_windows = build_sliding_windows(total_duration_s, args.sliding_window_sec, args.sliding_step_sec)

    video_stem = video_path.stem
    out_dir = Path(args.output_dir).expanduser().resolve() / video_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    event_fields = [
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
    sliding_fields = ["dindows_id"] + [f for f in event_fields if f != "window_id"]

    sliding_out: list[dict[str, Any]] = []
    for w in sliding_windows:
        agg = aggregate_rows_in_window(frame_rows, w)
        sliding_out.append(
            {
                "dindows_id": w.label,
                "time_period": w.time_period,
                "start_s": round(w.start_s, 3),
                "end_s": round(w.end_s, 3),
                **{k: round(v, 6) for k, v in agg.items()},
            }
        )

    event_out: list[dict[str, Any]] = []
    for w in event_windows:
        agg = aggregate_rows_in_window(frame_rows, w)
        event_out.append(
            {
                "window_id": w.label,
                "time_period": w.time_period,
                "start_s": round(w.start_s, 3),
                "end_s": round(w.end_s, 3),
                **{k: round(v, 6) for k, v in agg.items()},
            }
        )

    sliding_path = out_dir / f"{video_stem}_social_features_sliding_40s_1s.csv"
    event_path = out_dir / f"{video_stem}_social_features_events.csv"
    write_csv(sliding_path, sliding_out, sliding_fields)
    write_csv(event_path, event_out, event_fields)

    print(f"Done. Sliding social features: {sliding_path}")
    print(f"Done. Event social features: {event_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
