#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


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


def read_windows(segmentation_csv: Path) -> list[TimeWindow]:
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
        raise RuntimeError(f"No valid windows in: {segmentation_csv}")
    return windows


def build_sliding_windows(total_duration_s: float, window_s: float, step_s: float) -> list[TimeWindow]:
    if total_duration_s <= 0 or window_s <= 0 or step_s <= 0:
        return []
    out: list[TimeWindow] = []
    i = 0
    start_s = 0.0
    while start_s < total_duration_s:
        end_s = min(total_duration_s, start_s + window_s)
        if end_s > start_s:
            out.append(TimeWindow(label=f"Window {i + 1}", start_s=start_s, end_s=end_s))
        if end_s >= total_duration_s:
            break
        i += 1
        start_s += step_s
    return out


def extract_audio(input_video: Path, output_wav: Path, sr: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found. Install it and ensure it is on PATH.") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def union_coverage(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    arr = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    cur_s, cur_e = arr[0]
    for s, e in arr[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return float(sum(max(0.0, e - s) for s, e in merged))


def run_pyannote_diarization(wav_path: Path, hf_token: str, model_name: str, use_cuda: bool) -> list[SpeakerSegment]:
    # Compatibility patch: some recent torchaudio builds removed list_audio_backends,
    # while speechbrain (used by pyannote pipelines) still calls it.
    try:
        import torchaudio  # type: ignore

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise RuntimeError("pyannote.audio is required. Install with: pip install pyannote.audio") from exc

    try:
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
    except TypeError:
        # Backward compatibility for older pyannote versions.
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
    if use_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
        except Exception:
            pass

    diarization = pipeline(str(wav_path))
    if hasattr(diarization, "itertracks"):
        annotation = diarization
    elif hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
    elif hasattr(diarization, "to_annotation"):
        annotation = diarization.to_annotation()
    else:
        raise RuntimeError(f"Unsupported pyannote diarization output type: {type(diarization)!r}")

    out: list[SpeakerSegment] = []
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if end > start:
            out.append(SpeakerSegment(start=start, end=end, speaker=str(speaker)))
    return out


def compute_turn_taking_rate(spk_segments: list[SpeakerSegment], win: TimeWindow) -> float:
    if len(spk_segments) < 2 or win.duration <= 0:
        return 0.0
    arr = sorted(spk_segments, key=lambda x: (x.start, x.end))
    changes = 0
    prev = arr[0].speaker
    for s in arr[1:]:
        if s.speaker != prev:
            changes += 1
            prev = s.speaker
    return float(changes / max(1e-8, win.duration / 60.0))


def compute_overlap_speech(spk_segments: list[SpeakerSegment], win: TimeWindow) -> float:
    if not spk_segments or win.duration <= 0:
        return 0.0

    events: list[tuple[float, int]] = []
    for s in spk_segments:
        a = max(win.start_s, s.start)
        b = min(win.end_s, s.end)
        if b <= a:
            continue
        events.append((a, +1))
        events.append((b, -1))

    if not events:
        return 0.0

    events.sort(key=lambda x: (x[0], -x[1]))
    active = 0
    prev_t = events[0][0]
    overlap = 0.0
    for t, d in events:
        if t > prev_t and active >= 2:
            overlap += (t - prev_t)
        active += d
        prev_t = t

    return float(overlap / max(1e-8, win.duration))


def run_opensmile_extract(wav_path: Path, output_csv: Path, config_path: Path, opensmile_bin: str) -> None:
    cmd = [
        opensmile_bin,
        "-C",
        str(config_path),
        "-I",
        str(wav_path),
        "-lldcsvoutput",
        str(output_csv),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"openSMILE executable not found: {opensmile_bin}. "
            "Pass --opensmile-bin or set OPENSMILE_BIN."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"openSMILE failed: {' '.join(cmd)}\n{proc.stderr.strip()}")


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == "_")


def _find_key(header: list[str], candidates: list[str]) -> str | None:
    norm_to_raw = {_normalize_name(h): h for h in header}
    for c in candidates:
        k = _normalize_name(c)
        if k in norm_to_raw:
            return norm_to_raw[k]

    lowered = [(h, _normalize_name(h)) for h in header]
    for h, hn in lowered:
        if any(_normalize_name(c) in hn for c in candidates):
            return h
    return None


def _detect_delimiter(first_line: str) -> str:
    counts = {",": first_line.count(","), ";": first_line.count(";"), "\t": first_line.count("\t")}
    return max(counts, key=counts.get)


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_opensmile_features(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        first_line = f.readline()
        if not first_line:
            raise RuntimeError("openSMILE output is empty")
        delim = _detect_delimiter(first_line)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        header = reader.fieldnames or []

        time_col = _find_key(header, ["frameTime", "frame_time", "timestamp", "time"])
        loud_col = _find_key(header, ["loudness", "pcm_RMSenergy", "rms"])
        f0_col = _find_key(header, ["F0semitoneFrom27.5Hz", "F0", "pitch"])
        jitter_col = _find_key(header, ["jitterLocal", "jitter"])
        shimmer_col = _find_key(header, ["shimmerLocaldB", "shimmer"])

        if not time_col or not loud_col or not f0_col:
            raise RuntimeError("openSMILE output missing required columns: time/loudness/f0")

        time_vals: list[float] = []
        loud_vals: list[float] = []
        f0_vals: list[float] = []
        jitter_vals: list[float] = []
        shimmer_vals: list[float] = []

        for row in reader:
            time_vals.append(_to_float(row.get(time_col, "")))
            loud_vals.append(_to_float(row.get(loud_col, "")))
            f0_vals.append(_to_float(row.get(f0_col, "")))
            jitter_vals.append(_to_float(row.get(jitter_col, "")) if jitter_col else float("nan"))
            shimmer_vals.append(_to_float(row.get(shimmer_col, "")) if shimmer_col else float("nan"))

    arr = {
        "time_s": np.asarray(time_vals, dtype=np.float64),
        "loudness": np.asarray(loud_vals, dtype=np.float64),
        "f0": np.asarray(f0_vals, dtype=np.float64),
        "jitter": np.asarray(jitter_vals, dtype=np.float64),
        "shimmer": np.asarray(shimmer_vals, dtype=np.float64),
    }

    valid = np.isfinite(arr["time_s"])
    for k in arr:
        arr[k] = arr[k][valid]

    if arr["time_s"].size < 2 or float(np.nanmax(arr["time_s"]) - np.nanmin(arr["time_s"])) < 0.1:
        raise RuntimeError(
            "openSMILE output does not look like frame-level LLD (time span too short). "
            "Please check config/output mode."
        )
    return arr


def _robust_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.mean(x))


def _z_values(global_x: np.ndarray, window_x: np.ndarray) -> np.ndarray:
    gx = global_x[np.isfinite(global_x)]
    wx = window_x.copy()
    if gx.size == 0:
        return np.zeros_like(wx, dtype=np.float64)
    mu = float(np.mean(gx))
    sd = float(np.std(gx))
    if sd <= 1e-8:
        return np.zeros_like(wx, dtype=np.float64)
    z = (wx - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z


def compute_prosody_arousal_from_opensmile(feat: dict[str, np.ndarray], win: TimeWindow) -> tuple[float, float, float, float]:
    t = feat["time_s"]
    mask = (t >= win.start_s) & (t < win.end_s)
    if not np.any(mask):
        return 0.0, 0.0, 0.0, 0.0

    loud_w = feat["loudness"][mask]
    f0_w = feat["f0"][mask]
    jitter_w = feat["jitter"][mask]
    shimmer_w = feat["shimmer"][mask]

    loud_mean = _robust_mean(loud_w)
    f0_pos = f0_w[(np.isfinite(f0_w)) & (f0_w > 0)]
    f0_std = float(np.std(f0_pos)) if f0_pos.size else 0.0
    jitter_mean = _robust_mean(jitter_w)

    z_loud = _z_values(feat["loudness"], loud_w)
    z_f0 = _z_values(feat["f0"], f0_w)
    z_jitter = _z_values(feat["jitter"], jitter_w)
    z_shimmer = _z_values(feat["shimmer"], shimmer_w)

    arousal_raw = float(
        0.45 * np.mean(z_loud)
        + 0.30 * np.mean(np.abs(z_f0))
        + 0.15 * np.mean(np.abs(z_jitter))
        + 0.10 * np.mean(np.abs(z_shimmer))
    )
    if not math.isfinite(arousal_raw):
        arousal_raw = 0.0

    return float(np.tanh(arousal_raw)), loud_mean, f0_std, jitter_mean


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute speech conversation features via pyannote + openSMILE.")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--segments-csv", required=True, help="Event segmentation CSV")
    parser.add_argument("--output-dir", default="outputs/dynamic_social", help="Output root")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")

    parser.add_argument("--hf-token", default="", help="Hugging Face token for pyannote model access")
    parser.add_argument("--pyannote-model", default="pyannote/speaker-diarization-3.1", help="pyannote diarization model")
    parser.add_argument("--pyannote-cuda", action="store_true", help="Use CUDA for pyannote if available")

    parser.add_argument(
        "--opensmile-bin",
        default="./_third_party/opensmile/build/progsrc/smilextract/SMILExtract",
        help="openSMILE binary path/name",
    )
    parser.add_argument(
        "--opensmile-config",
        default="./_third_party/opensmile/config/egemaps/v01a/eGeMAPSv01a.conf",
        help="Path to openSMILE config file (recommend eGeMAPS)",
    )
    parser.add_argument("--sliding-window-sec", type=float, default=40.0, help="Sliding window length in seconds")
    parser.add_argument("--sliding-step-sec", type=float, default=1.0, help="Sliding window step in seconds")
    parser.add_argument("--duration-sec", type=float, help="Optional total video duration in seconds for full sliding-window coverage")
    return parser.parse_args()


def compute_rows(
    windows: list[TimeWindow],
    diarization_segments: list[SpeakerSegment],
    smile_feat: dict[str, np.ndarray],
    label_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w in windows:
        spk_segs = [
            SpeakerSegment(start=max(s.start, w.start_s), end=min(s.end, w.end_s), speaker=s.speaker)
            for s in diarization_segments
            if overlap_len(s.start, s.end, w.start_s, w.end_s) > 0
        ]
        speech_intervals = [(s.start, s.end) for s in spk_segs]
        speech_cov = union_coverage(speech_intervals)

        speech_ratio = float(speech_cov / max(1e-8, w.duration)) if w.duration > 0 else 0.0
        speaker_count = float(len({s.speaker for s in spk_segs})) if spk_segs else 0.0
        turn_taking_rate = compute_turn_taking_rate(spk_segs, w)
        overlap_speech = compute_overlap_speech(spk_segs, w)
        prosody_arousal, loud_mean, f0_std, jitter_mean = compute_prosody_arousal_from_opensmile(smile_feat, w)

        rows.append(
            {
                label_key: w.label,
                "time_period": w.time_period,
                "speech_ratio": round(speech_ratio, 6),
                "speaker_count": round(speaker_count, 6),
                "turn_taking_rate": round(turn_taking_rate, 6),
                "overlap_speech": round(overlap_speech, 6),
                "prosody_arousal": round(prosody_arousal, 6),
                "loudness_mean": round(loud_mean, 6),
                "pitch_std": round(f0_std, 6),
                "jitter_mean": round(jitter_mean, 6),
                "diarization_mode": "pyannote",
                "prosody_mode": "opensmile",
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    seg_csv = Path(args.segments_csv).expanduser().resolve()
    windows = read_windows(seg_csv)

    if not args.hf_token:
        raise RuntimeError("--hf-token is required for pyannote.audio model access.")

    opensmile_config = Path(args.opensmile_config).expanduser().resolve()
    if not opensmile_config.exists():
        raise RuntimeError(f"openSMILE config not found: {opensmile_config}")

    with tempfile.TemporaryDirectory(prefix="speech_conv_pa_os_") as td:
        wav_path = Path(td) / "audio.wav"
        smile_csv = Path(td) / "opensmile_features.csv"

        extract_audio(video_path, wav_path, sr=args.sample_rate)
        diarization_segments = run_pyannote_diarization(
            wav_path=wav_path,
            hf_token=args.hf_token,
            model_name=args.pyannote_model,
            use_cuda=args.pyannote_cuda,
        )
        run_opensmile_extract(
            wav_path=wav_path,
            output_csv=smile_csv,
            config_path=opensmile_config,
            opensmile_bin=args.opensmile_bin,
        )
        smile_feat = load_opensmile_features(smile_csv)

    event_rows = compute_rows(windows, diarization_segments, smile_feat, label_key="event")

    out_dir = Path(args.output_dir).expanduser().resolve() / video_path.stem
    out_path = out_dir / f"{video_path.stem}_speech_conversation_features.csv"
    write_csv(
        out_path,
        event_rows,
        [
            "event",
            "time_period",
            "speech_ratio",
            "speaker_count",
            "turn_taking_rate",
            "overlap_speech",
            "prosody_arousal",
            "loudness_mean",
            "pitch_std",
            "jitter_mean",
            "diarization_mode",
            "prosody_mode",
        ],
    )

    total_duration_s = float(args.duration_sec) if args.duration_sec else max(w.end_s for w in windows)
    sliding_windows = build_sliding_windows(total_duration_s, args.sliding_window_sec, args.sliding_step_sec)
    if sliding_windows:
        sliding_rows = compute_rows(sliding_windows, diarization_segments, smile_feat, label_key="window")
        sliding_path = out_dir / (
            f"{video_path.stem}_speech_conversation_features_sliding_"
            f"{int(args.sliding_window_sec)}s_{int(args.sliding_step_sec)}s.csv"
        )
        write_csv(
            sliding_path,
            sliding_rows,
            [
                "window",
                "time_period",
                "speech_ratio",
                "speaker_count",
                "turn_taking_rate",
                "overlap_speech",
                "prosody_arousal",
                "loudness_mean",
                "pitch_std",
                "jitter_mean",
                "diarization_mode",
                "prosody_mode",
            ],
        )
        print(f"Done. Sliding features: {sliding_path}")

    print(f"Done. Event features: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
