#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from linguistic_features import (
    heuristic_syntactic_complexity,
    lm_surprisal_predictability,
    mattr,
    read_event_text_rows,
    spacy,
    spacy_syntactic_complexity,
    tokenize,
    ttr,
    word_frequency_stats,
    word_length_stats,
    write_csv,
    zipf_frequency,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute event-only linguistic features.")
    p.add_argument("--event-csv", required=True, help="Event-level text CSV (e.g. *_text_channel_sentiment.csv)")
    p.add_argument("--lmstudio-base-url", default="http://127.0.0.1:1234")
    p.add_argument(
        "--llm-model",
        required=True,
        help="Exact LM Studio model identifier; recorded in the output CSV",
    )
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--output-dir", default="outputs/dynamic_social_4second")
    p.add_argument(
        "--allow-llm-errors",
        action="store_true",
        help="Write zero LLM scores plus lm_error instead of stopping on an LLM failure",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    event_csv = Path(args.event_csv).expanduser().resolve()
    event_rows = read_event_text_rows(event_csv)

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

    out: list[dict[str, Any]] = []
    for w in event_rows:
        toks = tokenize(w.text)
        token_count = len(toks)
        unique_count = len(set(toks)) if toks else 0

        ttr_v = ttr(toks)
        mattr_v = mattr(toks, 25)
        mean_len, std_len = word_length_stats(toks)
        mean_zipf, low_ratio, very_low_ratio = word_frequency_stats(toks)

        if nlp is not None and w.text.strip():
            mean_sent_len, clause_density, dep_depth = spacy_syntactic_complexity(nlp, w.text, toks)
        else:
            mean_sent_len, clause_density, dep_depth = heuristic_syntactic_complexity(w.text, toks)

        lm_error = ""
        lm_note = ""
        try:
            mean_surprisal, predictability, lm_note = lm_surprisal_predictability(
                w.text,
                base_url=args.lmstudio_base_url,
                model=args.llm_model,
                timeout_s=args.timeout,
            )
        except Exception as exc:
            if not args.allow_llm_errors:
                raise RuntimeError(f"LLM analysis failed for {w.window_id}: {exc}") from exc
            mean_surprisal, predictability = 0.0, 0.0
            lm_error = str(exc)[:300]

        out.append(
            {
                "window_type": w.window_type,
                "window_id": w.window_id,
                "time_period": w.time_period,
                "text": w.text,
                "token_count": token_count,
                "unique_token_count": unique_count,
                "ttr": round(ttr_v, 6),
                "mattr_25": round(mattr_v, 6),
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

    expected_suffix = "_text_channel_sentiment.csv"
    movie_name = (
        event_csv.name[: -len(expected_suffix)]
        if event_csv.name.endswith(expected_suffix)
        else event_csv.stem
    )
    out_dir = Path(args.output_dir).expanduser().resolve() / movie_name
    out_path = out_dir / f"{movie_name}_linguistic_features_events.csv"
    write_csv(
        out_path,
        out,
        [
            "window_type",
            "window_id",
            "time_period",
            "text",
            "token_count",
            "unique_token_count",
            "ttr",
            "mattr_25",
            "mean_word_length",
            "std_word_length",
            "mean_zipf_frequency",
            "low_frequency_ratio",
            "very_low_frequency_ratio",
            "wordfreq_available",
            "mean_sentence_length",
            "clause_density",
            "dep_tree_depth_mean",
            "mean_surprisal",
            "predictability_score",
            "llm_model",
            "lm_note",
            "lm_error",
        ],
    )
    print(f"Done. Event linguistic features: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
