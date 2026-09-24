#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from linguistic_features import (
    heuristic_syntactic_complexity,
    lm_surprisal_predictability,
    mattr,
    spacy,
    spacy_syntactic_complexity,
    tokenize,
    ttr,
    word_frequency_stats,
    word_length_stats,
    write_csv,
    zipf_frequency,
)

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None


def format_seconds(seconds: int) -> str:
    sec = max(0, int(seconds))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def extract_audio(video_path: Path, wav_path: Path, sample_rate: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found. Install it and ensure it is on PATH.") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def wav_duration_seconds(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        return float(wav.getnframes() / rate) if rate > 0 else 0.0


def transcribe_words_per_second(
    wav_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str,
) -> dict[int, str]:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is required. Install requirements.txt.")

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(wav_path),
        language=language or None,
        vad_filter=True,
        word_timestamps=True,
    )

    buckets: dict[int, list[str]] = {}
    for segment in segments:
        words = list(segment.words or [])
        if words:
            for word in words:
                start = float(word.start if word.start is not None else segment.start)
                end = float(word.end if word.end is not None else start)
                second = max(0, int(math.floor((start + end) / 2.0)))
                token = str(word.word or "").strip()
                if token:
                    buckets.setdefault(second, []).append(token)
        else:
            text = str(segment.text or "").strip()
            if text:
                second = max(0, int(math.floor(float(segment.start))))
                buckets.setdefault(second, []).append(text)

    return {second: " ".join(parts).strip() for second, parts in buckets.items()}


def write_transcript(path: Path, duration_s: int, per_second: dict[int, str]) -> None:
    rows = [
        {
            "second": second,
            "time_period": f"{format_seconds(second)}-{format_seconds(second + 1)}",
            "text": per_second.get(second, ""),
        }
        for second in range(duration_s)
    ]
    write_csv(path, rows, ["second", "time_period", "text"])


def read_transcript(path: Path) -> tuple[dict[int, str], int]:
    per_second: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            second = int(float(row.get("second", 0)))
            per_second[second] = str(row.get("text", ""))
    duration_s = max(per_second, default=-1) + 1
    return per_second, duration_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a video and compute per-second linguistic features."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output-dir", default="outputs/linguistic_features", help="Output root")
    parser.add_argument("--duration-sec", type=int, help="Number of output seconds; defaults to ceil(audio duration)")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--asr-model", default="small", help="faster-whisper model name")
    parser.add_argument("--asr-device", default="cpu", help="faster-whisper device")
    parser.add_argument("--asr-compute-type", default="int8", help="faster-whisper compute type")
    parser.add_argument("--language", default="en", help="ASR language code; empty enables detection")
    parser.add_argument("--force-transcribe", action="store_true", help="Ignore and replace cached transcript CSV")
    parser.add_argument(
        "--transcribe-only",
        action="store_true",
        help="Create/reuse the per-second transcript and stop before linguistic analysis",
    )
    parser.add_argument("--lmstudio-base-url", default="http://127.0.0.1:1234")
    parser.add_argument(
        "--llm-model",
        default="",
        help="Exact LM Studio model identifier; required unless --transcribe-only",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--allow-llm-errors",
        action="store_true",
        help="Write zero LLM scores plus lm_error instead of stopping on an LLM failure",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise RuntimeError(f"Video not found: {video_path}")

    out_dir = Path(args.output_dir).expanduser().resolve() / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / f"{video_path.stem}_transcript_per_second.csv"

    if transcript_path.exists() and not args.force_transcribe:
        print(f"Reusing transcript: {transcript_path}")
        per_second, inferred_duration_s = read_transcript(transcript_path)
    else:
        with tempfile.TemporaryDirectory(prefix="linguistic_per_second_") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            extract_audio(video_path, wav_path, args.sample_rate)
            inferred_duration_s = int(math.ceil(wav_duration_seconds(wav_path)))
            per_second = transcribe_words_per_second(
                wav_path=wav_path,
                model_name=args.asr_model,
                device=args.asr_device,
                compute_type=args.asr_compute_type,
                language=args.language,
            )

    duration_s = int(args.duration_sec) if args.duration_sec is not None else inferred_duration_s
    if duration_s < 0:
        raise RuntimeError("duration-sec must be non-negative")
    write_transcript(transcript_path, duration_s, per_second)
    print(f"Transcript ready: {transcript_path}")
    if args.transcribe_only:
        return 0
    if not args.llm_model:
        raise RuntimeError("--llm-model is required for linguistic analysis.")

    nlp: Any = None
    if spacy is not None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None
    if nlp is None:
        print("[warn] spaCy en_core_web_sm unavailable; using heuristic syntax features.")
    if zipf_frequency is None:
        print("[warn] wordfreq unavailable; word-frequency fields will be zero.")

    rows: list[dict[str, Any]] = []
    for second in range(duration_s):
        text = per_second.get(second, "")
        tokens = tokenize(text)
        mean_len, std_len = word_length_stats(tokens)
        mean_zipf, low_ratio, very_low_ratio = word_frequency_stats(tokens)
        if nlp is not None and text.strip():
            mean_sent_len, clause_density, dep_depth = spacy_syntactic_complexity(nlp, text, tokens)
        else:
            mean_sent_len, clause_density, dep_depth = heuristic_syntactic_complexity(text, tokens)

        lm_error = ""
        lm_note = ""
        try:
            mean_surprisal, predictability, lm_note = lm_surprisal_predictability(
                text,
                base_url=args.lmstudio_base_url,
                model=args.llm_model,
                timeout_s=args.timeout,
            )
        except Exception as exc:
            if not args.allow_llm_errors:
                raise RuntimeError(f"LLM analysis failed for second {second}: {exc}") from exc
            mean_surprisal, predictability = 0.0, 0.0
            lm_error = str(exc)[:300]

        rows.append(
            {
                "second": second,
                "time_period": f"{format_seconds(second)}-{format_seconds(second + 1)}",
                "text": text,
                "token_count": len(tokens),
                "unique_token_count": len(set(tokens)),
                "ttr": round(ttr(tokens), 6),
                "mattr_25": round(mattr(tokens, 25), 6),
                "mean_word_length": round(mean_len, 6),
                "std_word_length": round(std_len, 6),
                "mean_zipf_frequency": round(mean_zipf, 6),
                "low_frequency_ratio": round(low_ratio, 6),
                "very_low_frequency_ratio": round(very_low_ratio, 6),
                "wordfreq_available": 1 if zipf_frequency is not None else 0,
                "mean_sentence_length": round(mean_sent_len, 6),
                "clause_density": round(clause_density, 6),
                "dep_tree_depth_mean": round(dep_depth, 6),
                "mean_surprisal": round(mean_surprisal, 6),
                "predictability_score": round(predictability, 6),
                "llm_model": args.llm_model,
                "lm_note": lm_note,
                "lm_error": lm_error,
            }
        )

    output_path = out_dir / f"{video_path.stem}_linguistic_features_per_second.csv"
    write_csv(output_path, rows, list(rows[0].keys()) if rows else ["second", "time_period", "text"])
    print(f"Done. Per-second linguistic features: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
