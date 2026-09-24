#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import tempfile
import wave
from pathlib import Path

from speech_conversation_features import (
    SpeakerSegment,
    TimeWindow,
    compute_prosody_arousal_from_opensmile,
    extract_audio,
    load_opensmile_features,
    overlap_len,
    run_opensmile_extract,
    run_pyannote_diarization,
    write_csv,
)


def format_seconds(seconds: int) -> str:
    sec = max(0, int(seconds))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def get_wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate <= 0:
        return 0.0
    return float(frames / rate)


def has_speech_in_second(diarization_segments: list[SpeakerSegment], second_idx: int) -> int:
    start_s = float(second_idx)
    end_s = float(second_idx + 1)
    for seg in diarization_segments:
        if overlap_len(seg.start, seg.end, start_s, end_s) > 0:
            return 1
    return 0


def speaker_count_in_second(diarization_segments: list[SpeakerSegment], second_idx: int) -> int:
    start_s = float(second_idx)
    end_s = float(second_idx + 1)
    speakers = {
        seg.speaker
        for seg in diarization_segments
        if overlap_len(seg.start, seg.end, start_s, end_s) > 0
    }
    return len(speakers)


def overlap_speech_ratio_in_second(diarization_segments: list[SpeakerSegment], second_idx: int) -> float:
    start_s = float(second_idx)
    end_s = float(second_idx + 1)
    events: list[tuple[float, int]] = []
    for seg in diarization_segments:
        a = max(start_s, seg.start)
        b = min(end_s, seg.end)
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
    return float(overlap / max(1e-8, end_s - start_s))


def turn_change_in_second(diarization_segments: list[SpeakerSegment], second_idx: int) -> int:
    start_s = float(second_idx)
    end_s = float(second_idx + 1)
    arr = sorted(diarization_segments, key=lambda x: (x.start, x.end))
    for prev, cur in zip(arr, arr[1:]):
        if prev.speaker == cur.speaker:
            continue
        if start_s <= cur.start < end_s:
            return 1
    return 0


def build_second_rows(
    total_duration_s: int,
    diarization_segments: list[SpeakerSegment],
    smile_feat: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for second_idx in range(max(0, total_duration_s)):
        win = TimeWindow(label=f"S{second_idx:05d}", start_s=float(second_idx), end_s=float(second_idx + 1))
        prosody_arousal, loud_mean, pitch_std, jitter_mean = compute_prosody_arousal_from_opensmile(smile_feat, win)
        speech = has_speech_in_second(diarization_segments, second_idx)
        speaker_count = speaker_count_in_second(diarization_segments, second_idx)
        overlap_ratio = overlap_speech_ratio_in_second(diarization_segments, second_idx)
        turn_change = turn_change_in_second(diarization_segments, second_idx)
        prosody_valid = 1 if (speech == 1 and (loud_mean > 0.01 or pitch_std > 0.0 or jitter_mean > 0.0)) else 0
        rows.append(
            {
                "second": second_idx,
                "timestamp": format_seconds(second_idx),
                "speech": speech,
                "speaker_count_second": speaker_count,
                "overlap_speech_second": round(overlap_ratio, 6),
                "turn_change_second": turn_change,
                "prosody_valid": prosody_valid,
                "prosody_arousal": round(prosody_arousal, 6),
                "loudness_mean": round(loud_mean, 6),
                "pitch_std": round(pitch_std, 6),
                "jitter_mean": round(jitter_mean, 6),
                "diarization_mode": "pyannote",
                "prosody_mode": "opensmile",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    local_opensmile = repo_dir / "_third_party" / "opensmile"
    parent_opensmile = repo_dir.parent / "_third_party" / "opensmile"
    default_opensmile = local_opensmile if local_opensmile.exists() else parent_opensmile
    parser = argparse.ArgumentParser(description="Compute per-second speech/prosody features via pyannote + openSMILE.")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output-dir", default="outputs/dynamic_social", help="Output root")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--duration-sec", type=int, help="Optional total duration in seconds")

    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token; defaults to the HF_TOKEN environment variable",
    )
    parser.add_argument("--pyannote-model", default="pyannote/speaker-diarization-3.1", help="pyannote diarization model")
    parser.add_argument("--pyannote-cuda", action="store_true", help="Use CUDA for pyannote if available")

    parser.add_argument(
        "--opensmile-bin",
        default=os.environ.get(
            "OPENSMILE_BIN",
            str(default_opensmile / "build" / "progsrc" / "smilextract" / "SMILExtract"),
        ),
        help="openSMILE binary; defaults to the repository-local installation",
    )
    parser.add_argument(
        "--opensmile-config",
        default=os.environ.get(
            "OPENSMILE_CONFIG",
            str(default_opensmile / "config" / "egemaps" / "v01a" / "eGeMAPSv01a.conf"),
        ),
        help="eGeMAPS config; defaults to the repository-local installation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.hf_token:
        raise RuntimeError("Set HF_TOKEN or pass --hf-token for pyannote model access.")

    video_path = Path(args.video).expanduser().resolve()
    opensmile_config = Path(args.opensmile_config).expanduser().resolve()
    if not opensmile_config.exists():
        raise RuntimeError(f"openSMILE config not found: {opensmile_config}")

    with tempfile.TemporaryDirectory(prefix="speech_conv_per_sec_") as td:
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
        inferred_duration_s = int(math.ceil(get_wav_duration_seconds(wav_path)))

    total_duration_s = int(args.duration_sec) if args.duration_sec is not None else inferred_duration_s
    rows = build_second_rows(total_duration_s, diarization_segments, smile_feat)

    out_dir = Path(args.output_dir).expanduser().resolve() / video_path.stem
    out_path = out_dir / f"{video_path.stem}_speech_conversation_features_per_second.csv"
    write_csv(
        out_path,
        rows,
        [
            "second",
            "timestamp",
            "speech",
            "speaker_count_second",
            "overlap_speech_second",
            "turn_change_second",
            "prosody_valid",
            "prosody_arousal",
            "loudness_mean",
            "pitch_std",
            "jitter_mean",
            "diarization_mode",
            "prosody_mode",
        ],
    )
    print(f"Done. Per-second speech features: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
