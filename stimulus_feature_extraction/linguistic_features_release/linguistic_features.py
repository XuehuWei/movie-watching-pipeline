#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

try:
    from wordfreq import zipf_frequency
except Exception:  # pragma: no cover
    zipf_frequency = None

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None


TOKEN_RE = re.compile(r"[A-Za-z']+")
SENT_SPLIT_RE = re.compile(r"[.!?]+")


@dataclass
class WindowText:
    window_id: str
    time_period: str
    text: str
    window_type: str  # event|sliding


def format_seconds(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return float(len(set(tokens)) / len(tokens))


def mattr(tokens: list[str], w: int = 25) -> float:
    if not tokens:
        return 0.0
    if len(tokens) <= w:
        return ttr(tokens)
    vals = []
    for i in range(0, len(tokens) - w + 1):
        vals.append(len(set(tokens[i : i + w])) / w)
    return float(sum(vals) / len(vals)) if vals else 0.0


def word_length_stats(tokens: list[str]) -> tuple[float, float]:
    if not tokens:
        return 0.0, 0.0
    lens = [len(t) for t in tokens]
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    return float(mean), float(math.sqrt(var))


def word_frequency_stats(tokens: list[str]) -> tuple[float, float, float]:
    if not tokens:
        return 0.0, 0.0, 0.0
    if zipf_frequency is None:
        return 0.0, 0.0, 0.0
    vals = [float(zipf_frequency(t, "en")) for t in tokens]
    mean_zipf = float(sum(vals) / len(vals)) if vals else 0.0
    low_freq_ratio = float(sum(1 for v in vals if v < 3.0) / len(vals)) if vals else 0.0
    very_low_ratio = float(sum(1 for v in vals if v < 2.0) / len(vals)) if vals else 0.0
    return mean_zipf, low_freq_ratio, very_low_ratio


def heuristic_syntactic_complexity(text: str, tokens: list[str]) -> tuple[float, float, float]:
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text or "") if s.strip()]
    if not sents:
        sents = [text.strip()] if text.strip() else []
    if not sents:
        return 0.0, 0.0, 0.0

    sent_lens = [len(tokenize(s)) for s in sents]
    mean_sent_len = float(sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0

    clause_markers = {
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "if",
        "because",
        "although",
        "though",
        "while",
        "when",
        "since",
        "unless",
        "whereas",
        "before",
        "after",
    }
    n_clauses = sum(1 for t in tokens if t in clause_markers)
    clause_density = float(n_clauses / max(1, len(sents)))

    # Heuristic fallback for dependency depth when spaCy is unavailable.
    dep_depth_proxy = float(min(10.0, mean_sent_len / 4.0))
    return mean_sent_len, clause_density, dep_depth_proxy


def spacy_syntactic_complexity(nlp: Any, text: str, tokens: list[str]) -> tuple[float, float, float]:
    doc = nlp(text)
    sents = list(doc.sents)
    if not sents:
        return heuristic_syntactic_complexity(text, tokens)

    sent_lens = [sum(1 for t in s if t.is_alpha) for s in sents]
    mean_sent_len = float(sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0

    clause_labels = {"ccomp", "xcomp", "advcl", "relcl", "acl", "csubj"}
    n_clauses = sum(1 for t in doc if t.dep_ in clause_labels)
    clause_density = float(n_clauses / max(1, len(sents)))

    def depth(tok: Any) -> int:
        d = 1
        cur = tok
        while cur.head != cur:
            d += 1
            if d > 50:
                break
            cur = cur.head
        return d

    depths = [depth(t) for t in doc if t.is_alpha]
    dep_depth_mean = float(sum(depths) / len(depths)) if depths else 0.0
    return mean_sent_len, clause_density, dep_depth_mean


def parse_json_block(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL | re.IGNORECASE)
        if m:
            t = m.group(1).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        t = m.group(0)
    obj = json.loads(t)
    if not isinstance(obj, dict):
        raise RuntimeError("LLM response is not a JSON object")
    return obj


def lm_surprisal_predictability(text: str, base_url: str, model: str, timeout_s: int) -> tuple[float, float, str]:
    if not text.strip():
        return 0.0, 0.0, "NO_TEXT"

    prompt = (
        "Estimate language uncertainty for this text. Return ONLY JSON with keys: "
        "mean_surprisal, predictability_score, note. "
        "Constraints: mean_surprisal in [0,20], predictability_score in [0,1]. "
        "Higher surprisal means lower predictability. "
        f"Text: {text}"
    )

    def parse_raw(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("choices"), list) and payload["choices"]:
            return str(payload["choices"][0].get("message", {}).get("content", ""))
        if isinstance(payload.get("output"), list) and payload["output"]:
            first = payload["output"][0]
            if isinstance(first, dict) and "content" in first:
                return str(first.get("content", ""))
        for k in ("output", "response", "text", "content"):
            if k in payload:
                return str(payload[k])
        raise RuntimeError(f"Unsupported LM response format: {payload}")

    raw: str | None = None
    last_exc: Exception | None = None

    body_native: dict[str, Any] = {
        "model": model,
        "system_prompt": "You output strict JSON only.",
        "input": prompt,
        "temperature": 0,
    }

    body_openai: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    # 1) Prefer LM Studio native endpoint because this workspace is configured against /api/v1/chat.
    try:
        url = base_url.rstrip("/") + "/api/v1/chat"
        resp = requests.post(url, json=body_native, timeout=timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text[:600]}")
        raw = parse_raw(resp.json())
    except Exception as exc:
        last_exc = exc

    # 2) OpenAI-compatible fallback for servers exposing /v1/chat/completions only.
    if raw is None:
        try:
            url = base_url.rstrip("/") + "/v1/chat/completions"
            resp = requests.post(url, json=body_openai, timeout=timeout_s)
            if resp.status_code >= 400:
                raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text[:600]}")
            raw = parse_raw(resp.json())
        except Exception as exc:
            if last_exc is not None:
                raise RuntimeError(f"LM Studio native error: {last_exc}; OpenAI endpoint error: {exc}") from exc
            raise

    data = parse_json_block(str(raw))

    surprisal = max(0.0, min(20.0, float(data.get("mean_surprisal", 0.0))))
    predictability = max(0.0, min(1.0, float(data.get("predictability_score", 0.0))))
    note = str(data.get("note", ""))[:220]
    return surprisal, predictability, note


def parse_time_period_to_seconds(tp: str) -> tuple[float, float]:
    # supports mm:ss-mm:ss and hh:mm:ss-hh:mm:ss
    a, b = [x.strip() for x in tp.split("-")]
    return parse_mmss(a), parse_mmss(b)


def read_event_text_rows(event_csv: Path) -> list[WindowText]:
    rows: list[WindowText] = []
    with event_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, 1):
            wid = str(row.get("event") or row.get("window_id") or f"Event {i}")
            tp = str(row.get("time_period") or row.get("timestamp") or "")
            txt = str(row.get("text") or "")
            if not tp:
                continue
            rows.append(WindowText(window_id=wid, time_period=tp, text=txt, window_type="event"))
    return rows


def read_per_second_text(per_second_csv: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with per_second_csv.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sec = int(float(row.get("second", 0)))
            out[sec] = str(row.get("text", ""))
    return out


def build_sliding_from_per_second(
    per_second: dict[int, str], win_s: int, step_s: int, duration_s: int | None = None
) -> list[WindowText]:
    if not per_second and duration_s is None:
        return []
    mx = max(per_second.keys()) if per_second else -1
    total_s = max(mx + 1, int(duration_s or 0))
    out: list[WindowText] = []
    idx = 0
    start = 0
    while start + win_s <= total_s:
        end = start + win_s
        parts = [per_second.get(s, "") for s in range(start, end)]
        text = " ".join(x for x in parts if x.strip()).strip()
        out.append(
            WindowText(
                window_id=f"SW_{idx:05d}",
                time_period=f"{format_seconds(start)}-{format_seconds(end)}",
                text=text,
                window_type="sliding",
            )
        )
        idx += 1
        start += step_s
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute linguistic features per event and optional 40s sliding windows.")
    p.add_argument("--event-csv", required=True, help="Event-level text CSV (e.g. *_text_channel_sentiment.csv)")
    p.add_argument("--per-second-csv", help="Optional per-second text CSV for building sliding windows")
    p.add_argument("--sliding-window-sec", type=int, default=40)
    p.add_argument("--sliding-step-sec", type=int, default=1)
    p.add_argument("--lmstudio-base-url", default="http://127.0.0.1:1234")
    p.add_argument(
        "--llm-model",
        required=True,
        help="Exact LM Studio model identifier; recorded in the output CSV",
    )
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--llm-workers", type=int, default=4)
    p.add_argument("--duration-sec", type=int, help="Optional total video duration in seconds for full sliding-window coverage")
    p.add_argument("--output", default="outputs/dynamic/linguistic_features.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    event_rows = read_event_text_rows(Path(args.event_csv).expanduser().resolve())
    all_rows = list(event_rows)

    if args.per_second_csv:
        per_second = read_per_second_text(Path(args.per_second_csv).expanduser().resolve())
        all_rows.extend(
            build_sliding_from_per_second(
                per_second,
                args.sliding_window_sec,
                args.sliding_step_sec,
                duration_s=args.duration_sec,
            )
        )

    nlp = None
    if spacy is not None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None

    out: list[dict[str, Any]] = []
    texts_for_llm: set[str] = set()
    for w in all_rows:
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

        if w.text.strip():
            texts_for_llm.add(w.text)

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
                "mean_surprisal": 0.0,
                "predictability_score": 0.0,
                "llm_model": args.llm_model,
                "lm_note": "",
                "lm_error": "",
            }
        )

    llm_results: dict[str, tuple[float, float, str, str]] = {}
    if texts_for_llm:
        max_workers = max(1, int(args.llm_workers))

        def run_one(text: str) -> tuple[str, float, float, str, str]:
            try:
                mean_surprisal, predictability, note = lm_surprisal_predictability(
                    text,
                    base_url=args.lmstudio_base_url,
                    model=args.llm_model,
                    timeout_s=args.timeout,
                )
                return text, mean_surprisal, predictability, note, ""
            except Exception as exc:
                return text, 0.0, 0.0, "", str(exc)[:300]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, text) for text in sorted(texts_for_llm)]
            for future in as_completed(futures):
                text, mean_surprisal, predictability, note, error = future.result()
                llm_results[text] = (mean_surprisal, predictability, note, error)

    for row in out:
        text = str(row.get("text", ""))
        if not text.strip():
            row["lm_note"] = "NO_TEXT"
            continue
        mean_surprisal, predictability, note, error = llm_results.get(text, (0.0, 0.0, "", ""))
        row["mean_surprisal"] = round(mean_surprisal, 6)
        row["predictability_score"] = round(predictability, 6)
        row["lm_note"] = note
        row["lm_error"] = error

    fieldnames = [
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
    ]

    out_path = Path(args.output).expanduser().resolve()
    events_path = out_path.with_name(f"{out_path.stem}_events.csv")
    sliding_path = out_path.with_name(
        f"{out_path.stem}_sliding_{int(args.sliding_window_sec)}s_{int(args.sliding_step_sec)}s.csv"
    )

    event_out = [row for row in out if str(row.get("window_type", "")) == "event"]
    sliding_out = [row for row in out if str(row.get("window_type", "")) == "sliding"]

    write_csv(events_path, event_out, fieldnames)
    write_csv(sliding_path, sliding_out, fieldnames)

    print(f"Done. Linguistic features (events): {events_path}")
    print(f"Done. Linguistic features (sliding): {sliding_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
