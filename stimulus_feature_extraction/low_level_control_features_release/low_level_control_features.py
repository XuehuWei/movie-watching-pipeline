#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from pliers.extractors import (
        BrightnessExtractor,
        OnsetStrengthMultiExtractor,
        RMSExtractor,
        SpectralCentroidExtractor,
    )
    from pliers.filters import FrameSamplingFilter
    from pliers.stimuli import AudioStim, VideoStim
except Exception:  # pragma: no cover
    BrightnessExtractor = None
    OnsetStrengthMultiExtractor = None
    RMSExtractor = None
    SpectralCentroidExtractor = None
    FrameSamplingFilter = None
    AudioStim = None
    VideoStim = None


@dataclass
class TimeWindow:
    label: str
    start_s: float
    end_s: float

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
            if end_s > start_s:
                windows.append(TimeWindow(label, start_s, end_s))
    if not windows:
        raise RuntimeError(f"No valid event windows in: {segmentation_csv}")
    return windows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_audio(video_path: Path, wav_path: Path, sr: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def get_video_duration_and_fps(video_path: Path) -> tuple[float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        fps = 25.0
    duration_s = float(frame_count / fps) if frame_count > 0 else 0.0
    return duration_s, fps


def _infer_numeric_feature_column(df: Any, exclude: set[str]) -> str:
    candidates: list[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        try:
            if np.issubdtype(df[col].dtype, np.number):
                candidates.append(col)
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(f"Could not infer feature column from pliers result columns: {list(df.columns)}")
    return candidates[0]


def _extract_per_second_audio_feature(stim: Any, extractor: Any, value_col: str) -> list[float]:
    df = extractor.transform(stim).to_df()
    if df.empty:
        return []
    if value_col not in df.columns:
        value_col = _infer_numeric_feature_column(df, {"onset", "duration", "order", "object_id"})
    out: dict[int, list[float]] = {}
    for _, row in df.iterrows():
        sec = max(0, int(float(row.get("onset", 0.0))))
        value = np.asarray(row[value_col], dtype=float)
        out.setdefault(sec, []).append(float(np.mean(value)))
    max_sec = max(out.keys()) if out else -1
    return [float(np.mean(out.get(sec, [0.0]))) for sec in range(max_sec + 1)]


def compute_per_second_audio_features_with_pliers(wav_path: Path, sample_rate: int) -> dict[str, list[float]]:
    if (
        AudioStim is None
        or RMSExtractor is None
        or SpectralCentroidExtractor is None
        or OnsetStrengthMultiExtractor is None
    ):
        raise RuntimeError(
            "pliers is not installed in the current environment. Install it first, e.g. `pip install pliers`."
        )

    stim = AudioStim(filename=str(wav_path))
    return {
        "audio_rms": _extract_per_second_audio_feature(
            stim, RMSExtractor(hop_length=sample_rate), "rms"
        ),
        "spectral_centroid_mean": _extract_per_second_audio_feature(
            stim,
            SpectralCentroidExtractor(hop_length=sample_rate),
            "spectral_centroid",
        ),
        "onset_strength_mean": _extract_per_second_audio_feature(
            stim,
            OnsetStrengthMultiExtractor(hop_length=sample_rate),
            "onset_strength_multi",
        ),
    }


def compute_per_second_brightness_with_pliers(video_path: Path) -> dict[int, float]:
    if VideoStim is None or FrameSamplingFilter is None or BrightnessExtractor is None:
        raise RuntimeError(
            "pliers is not installed in the current environment. Install it first, e.g. `pip install pliers`."
        )

    stim = VideoStim(filename=str(video_path))
    sampled = FrameSamplingFilter(hertz=1).transform(stim)
    out: dict[int, float] = {}
    ext = BrightnessExtractor()
    results = ext.transform(sampled)
    if not isinstance(results, list):
        results = [results]
    for res in results:
        df = res.to_df()
        if df.empty:
            continue
        value_col = "brightness" if "brightness" in df.columns else _infer_numeric_feature_column(
            df, {"onset", "duration", "order"}
        )
        row = df.iloc[0]
        sec = max(0, int(float(row.get("onset", 0.0))))
        out[sec] = float(row[value_col])
    return out


def compute_per_second_visual_features(video_path: Path) -> tuple[list[dict[str, float]], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0:
        fps = 25.0
    duration_s = float(frame_count / fps) if frame_count > 0 else 0.0
    total_seconds = int(math.ceil(duration_s))

    rows = [
        {
            "frame_count": 0.0,
            "brightness_sum": 0.0,
            "contrast_sum": 0.0,
            "saturation_sum": 0.0,
            "motion_sum": 0.0,
            "optical_flow_sum": 0.0,
            "shot_cut_count": 0.0,
        }
        for _ in range(total_seconds)
    ]

    brightness_by_sec = compute_per_second_brightness_with_pliers(video_path)

    prev_gray: np.ndarray | None = None
    prev_flow_gray: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        flow_width = min(320, gray.shape[1])
        flow_height = max(1, int(round(gray.shape[0] * flow_width / gray.shape[1])))
        flow_gray = cv2.resize(gray, (flow_width, flow_height), interpolation=cv2.INTER_AREA)
        second_idx = min(total_seconds - 1, max(0, int(frame_idx / fps))) if total_seconds > 0 else 0
        if total_seconds > 0:
            rec = rows[second_idx]
            rec["frame_count"] += 1.0
            rec["brightness_sum"] += float(np.mean(gray) / 255.0)
            rec["contrast_sum"] += float(np.std(gray) / 255.0)
            rec["saturation_sum"] += float(np.mean(hsv[:, :, 1]) / 255.0)

            if prev_flow_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_flow_gray,
                    flow_gray,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0,
                )
                magnitude = cv2.magnitude(flow[:, :, 0], flow[:, :, 1])
                diagonal = math.hypot(flow_width, flow_height)
                rec["optical_flow_sum"] += float(np.mean(magnitude) / max(diagonal, 1.0))
            prev_flow_gray = flow_gray

            if prev_gray is not None:
                motion = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0)
                rec["motion_sum"] += motion

                hist_cur = cv2.calcHist([gray], [0], None, [64], [0, 256])
                if prev_hist is not None:
                    corr = cv2.compareHist(prev_hist, hist_cur, cv2.HISTCMP_CORREL)
                    if corr < 0.70:
                        rec["shot_cut_count"] += 1.0
                prev_hist = hist_cur
            else:
                prev_hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            prev_gray = gray
        frame_idx += 1

    cap.release()

    out: list[dict[str, float]] = []
    for sec_idx, rec in enumerate(rows):
        frame_n = max(1.0, rec["frame_count"])
        brightness_mean = float(brightness_by_sec.get(sec_idx, rec["brightness_sum"] / frame_n))
        out.append(
            {
                "second": float(sec_idx),
                "brightness_mean": brightness_mean,
                "contrast_std": rec["contrast_sum"] / frame_n,
                "saturation_mean": rec["saturation_sum"] / frame_n,
                "motion_energy": rec["motion_sum"] / frame_n,
                "optical_flow_mean": rec["optical_flow_sum"] / frame_n,
                "shot_cut_count": rec["shot_cut_count"],
                "visual_valid": 1.0 if rec["frame_count"] > 0 else 0.0,
            }
        )
    return out, duration_s


def merge_per_second_features(video_path: Path, sample_rate: int, duration_sec: int | None) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="llctrl_") as td:
        wav_path = Path(td) / "audio.wav"
        extract_audio(video_path, wav_path, sample_rate)
        audio_features = compute_per_second_audio_features_with_pliers(wav_path, sample_rate)

    visual_rows, visual_duration = compute_per_second_visual_features(video_path)
    inferred_seconds = max(
        *(len(values) for values in audio_features.values()),
        len(visual_rows),
        int(math.ceil(visual_duration)),
    )
    total_seconds = int(duration_sec) if duration_sec is not None else inferred_seconds

    rows: list[dict[str, Any]] = []
    for sec in range(total_seconds):
        v = visual_rows[sec] if sec < len(visual_rows) else None
        rows.append(
            {
                "second": sec,
                "timestamp": format_seconds(float(sec)),
                "visual_valid": int(v["visual_valid"]) if v is not None else 0,
                "motion_energy": round(float(v["motion_energy"]) if v is not None else 0.0, 6),
                "optical_flow_mean": round(float(v["optical_flow_mean"]) if v is not None else 0.0, 6),
                "brightness_mean": round(float(v["brightness_mean"]) if v is not None else 0.0, 6),
                "contrast_std": round(float(v["contrast_std"]) if v is not None else 0.0, 6),
                "saturation_mean": round(float(v["saturation_mean"]) if v is not None else 0.0, 6),
                "audio_rms": round(
                    float(audio_features["audio_rms"][sec])
                    if sec < len(audio_features["audio_rms"])
                    else 0.0,
                    6,
                ),
                "spectral_centroid_mean": round(
                    float(audio_features["spectral_centroid_mean"][sec])
                    if sec < len(audio_features["spectral_centroid_mean"])
                    else 0.0,
                    6,
                ),
                "onset_strength_mean": round(
                    float(audio_features["onset_strength_mean"][sec])
                    if sec < len(audio_features["onset_strength_mean"])
                    else 0.0,
                    6,
                ),
                "shot_cut_count": int(round(float(v["shot_cut_count"]))) if v is not None else 0,
            }
        )
    return rows


def aggregate_second_rows(rows: list[dict[str, Any]], window: TimeWindow) -> dict[str, float]:
    sel = [r for r in rows if window.start_s <= float(r["second"]) < window.end_s]
    if not sel:
        return {
            "motion_energy": 0.0,
            "optical_flow_mean": 0.0,
            "brightness_mean": 0.0,
            "contrast_std": 0.0,
            "saturation_mean": 0.0,
            "audio_rms": 0.0,
            "spectral_centroid_mean": 0.0,
            "onset_strength_mean": 0.0,
            "shot_cut_count": 0.0,
            "visual_valid_ratio": 0.0,
        }

    n = float(len(sel))
    return {
        "motion_energy": float(sum(float(r["motion_energy"]) for r in sel) / n),
        "optical_flow_mean": float(sum(float(r["optical_flow_mean"]) for r in sel) / n),
        "brightness_mean": float(sum(float(r["brightness_mean"]) for r in sel) / n),
        "contrast_std": float(sum(float(r["contrast_std"]) for r in sel) / n),
        "saturation_mean": float(sum(float(r["saturation_mean"]) for r in sel) / n),
        "audio_rms": float(sum(float(r["audio_rms"]) for r in sel) / n),
        "spectral_centroid_mean": float(sum(float(r["spectral_centroid_mean"]) for r in sel) / n),
        "onset_strength_mean": float(sum(float(r["onset_strength_mean"]) for r in sel) / n),
        "shot_cut_count": float(sum(float(r["shot_cut_count"]) for r in sel)),
        "visual_valid_ratio": float(sum(float(r["visual_valid"]) for r in sel) / n),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract low-level control features from movie video/audio.")
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--output-dir", default="outputs/dynamic_social", help="Output root directory")
    p.add_argument("--segments-csv", help="Optional event segmentation CSV. If provided, also write event-level features.")
    p.add_argument("--duration-sec", type=int, help="Optional total duration in seconds for per-second output.")
    p.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate for RMS extraction")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if (
        BrightnessExtractor is None
        or RMSExtractor is None
        or SpectralCentroidExtractor is None
        or OnsetStrengthMultiExtractor is None
        or FrameSamplingFilter is None
        or VideoStim is None
        or AudioStim is None
    ):
        raise RuntimeError(
            "This script now depends on pliers. Install it in the active environment first, e.g. `pip install pliers`."
        )
    video_path = Path(args.video).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    second_rows = merge_per_second_features(video_path, args.sample_rate, args.duration_sec)
    per_second_path = out_dir / f"{video_path.stem}_low_level_control_features_per_second.csv"
    write_csv(
        per_second_path,
        second_rows,
        [
            "second",
            "timestamp",
            "visual_valid",
            "motion_energy",
            "optical_flow_mean",
            "brightness_mean",
            "contrast_std",
            "saturation_mean",
            "audio_rms",
            "spectral_centroid_mean",
            "onset_strength_mean",
            "shot_cut_count",
        ],
    )

    if args.segments_csv:
        windows = read_event_windows(Path(args.segments_csv).expanduser().resolve())
        event_rows: list[dict[str, Any]] = []
        for w in windows:
            agg = aggregate_second_rows(second_rows, w)
            event_rows.append(
                {
                    "window_id": w.label,
                    "time_period": w.time_period,
                    "start_s": round(w.start_s, 3),
                    "end_s": round(w.end_s, 3),
                    "motion_energy": round(agg["motion_energy"], 6),
                    "optical_flow_mean": round(agg["optical_flow_mean"], 6),
                    "brightness_mean": round(agg["brightness_mean"], 6),
                    "contrast_std": round(agg["contrast_std"], 6),
                    "saturation_mean": round(agg["saturation_mean"], 6),
                    "audio_rms": round(agg["audio_rms"], 6),
                    "spectral_centroid_mean": round(agg["spectral_centroid_mean"], 6),
                    "onset_strength_mean": round(agg["onset_strength_mean"], 6),
                    "shot_cut_count": round(agg["shot_cut_count"], 6),
                    "visual_valid_ratio": round(agg["visual_valid_ratio"], 6),
                }
            )
        event_path = out_dir / f"{video_path.stem}_low_level_control_features_events.csv"
        write_csv(
            event_path,
            event_rows,
            [
                "window_id",
                "time_period",
                "start_s",
                "end_s",
                "motion_energy",
                "optical_flow_mean",
                "brightness_mean",
                "contrast_std",
                "saturation_mean",
                "audio_rms",
                "spectral_centroid_mean",
                "onset_strength_mean",
                "shot_cut_count",
                "visual_valid_ratio",
            ],
        )
        print(f"Done. Per-second control features: {per_second_path}")
        print(f"Done. Event control features: {event_path}")
    else:
        print(f"Done. Per-second control features: {per_second_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
