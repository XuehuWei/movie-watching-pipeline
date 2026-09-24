#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from speech_conversation_features import (
    compute_rows,
    extract_audio,
    load_opensmile_features,
    read_windows,
    run_opensmile_extract,
    run_pyannote_diarization,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    local_opensmile = repo_dir / "_third_party" / "opensmile"
    parent_opensmile = repo_dir.parent / "_third_party" / "opensmile"
    default_opensmile = local_opensmile if local_opensmile.exists() else parent_opensmile
    parser = argparse.ArgumentParser(description="Compute event-only speech conversation features via pyannote + openSMILE.")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--segments-csv", required=True, help="Event segmentation CSV")
    parser.add_argument("--output-dir", default="outputs/dynamic_social", help="Output root")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")

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
    video_path = Path(args.video).expanduser().resolve()
    seg_csv = Path(args.segments_csv).expanduser().resolve()
    windows = read_windows(seg_csv)

    if not args.hf_token:
        raise RuntimeError("Set HF_TOKEN or pass --hf-token for pyannote model access.")

    opensmile_config = Path(args.opensmile_config).expanduser().resolve()
    if not opensmile_config.exists():
        raise RuntimeError(f"openSMILE config not found: {opensmile_config}")

    with tempfile.TemporaryDirectory(prefix="speech_conv_pa_os_events_") as td:
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

    print(f"Done. Event features: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
