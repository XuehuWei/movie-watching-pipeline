# Per-Second and Event-Level Linguistic Feature Extraction

This repository computes per-second or event-level lexical, syntactic, and
LLM-estimated language uncertainty features.

The per-second entry point accepts a video and transcribes it with
faster-whisper. The event-level entry point accepts a prepared event-text CSV
and does not create event boundaries.

## Repository Files

```text
linguistic_features_events_only.py       # run this event-level entry point
linguistic_features_per_second.py         # run this video-to-per-second entry point
linguistic_features.py                   # shared feature functions
requirements.txt
.gitignore
README.md
example/event_text_example.csv
example/example_linguistic_features_events.csv
example/example_linguistic_features_per_second.csv
```

`linguistic_features.py` is imported automatically. It does not need to be run
before the event-only script.

## Software and Models

| Component | Required? | Purpose |
| --- | --- | --- |
| Python 3.10+ | Yes | Runs feature extraction and CSV input/output. |
| FFmpeg | Per-second only | Extracts mono 16 kHz audio from video. |
| faster-whisper | Per-second only | Transcribes speech with word-level timestamps. |
| requests | Yes | Sends local HTTP requests to LM Studio. |
| wordfreq | Recommended and installed by default | Computes English Zipf word-frequency statistics. |
| spaCy | Recommended and installed by default | Computes sentence and dependency-based syntax features. |
| `en_core_web_sm` | Recommended | English spaCy parser model. Without it, the script uses heuristic syntax features. |
| LM Studio | Yes for LLM fields | Serves a local language model through `http://127.0.0.1:1234`. |
| User-selected LM Studio model | Yes for LLM fields | Estimates the custom surprisal and predictability fields. The exact identifier is required through `--llm-model`. |

No Hugging Face token, OpenAI API key, pyannote, openSMILE, OpenFace, YOLO, or
MediaPipe installation is required. The first faster-whisper run may download
the selected ASR model if it is not already cached.

## Which Features Use the LLM?

Only these fields use LM Studio:

- `mean_surprisal`
- `predictability_score`
- `lm_note`
- `llm_model`
- `lm_error`

These fields are computed without an LLM:

- `token_count`
- `unique_token_count`
- `ttr`
- `mattr_25`
- `mean_word_length`
- `std_word_length`
- `mean_zipf_frequency`
- `low_frequency_ratio`
- `very_low_frequency_ratio`
- `mean_sentence_length`
- `clause_density`
- `dep_tree_depth_mean`

## Important Interpretation of Surprisal

`mean_surprisal` is a **prompt-based LLM estimate on a 0-20 scale**. It is not
token-level information-theoretic surprisal calculated from model logits or
negative log probabilities. `predictability_score` is an LLM estimate in
`[0,1]`.

These measures depend on the model, quantization, runtime, prompt, and model
version. They should be described as LLM-estimated language uncertainty and
should not be treated as equivalent to conventional token surprisal.

## Installation

The per-second pipeline requires FFmpeg. On macOS with Homebrew:

```bash
brew install ffmpeg
```

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

Create a clean Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Verify the Python dependencies:

```bash
python -c "import requests, spacy, wordfreq; spacy.load('en_core_web_sm'); print('Dependencies OK')"
python linguistic_features_events_only.py --help
python linguistic_features_per_second.py --help
```

## LM Studio Setup

1. Install and open [LM Studio](https://lmstudio.ai/).
2. Download a compatible text model, for example `google/gemma-4-12b`.
3. Open LM Studio's **Developer** tab.
4. Start the local API server on port `1234`.
5. Confirm that the model identifier shown by LM Studio matches the value passed
   to `--llm-model`.

The server can also be started from the LM Studio CLI:

```bash
lms server start --port 1234
```

Test the model before running the full CSV:

```bash
curl http://localhost:1234/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-12b",
    "system_prompt": "Return strict JSON only.",
    "input": "Return {\"mean_surprisal\": 5, \"predictability_score\": 0.8, \"note\": \"test\"}."
  }'
```

The script first calls LM Studio's native `/api/v1/chat` endpoint. If that
fails, it tries the OpenAI-compatible `/v1/chat/completions` endpoint.

The current script assumes that LM Studio authentication is disabled, which is
the default for a local server. Keep the server bound to localhost unless
remote access is intentionally configured and secured.

## Input CSV

The input CSV requires these columns:

| Column | Required? | Description |
| --- | --- | --- |
| `event` or `window_id` | Recommended | Event identifier. If absent, sequential labels are generated. |
| `time_period` or `timestamp` | Yes | Event interval, such as `00:00-00:08`. Rows without a time value are skipped. |
| `text` | Yes | English event-level dialogue or transcript text. |

Minimal example:

```csv
event,time_period,text
Event 0,00:00-00:08,"Where are we going?"
Event 1,00:08-00:18,"I do not know yet, but we need to leave now."
```

Additional columns such as sentiment labels are allowed and ignored.

## Per-Second Video Processing

Per-second extraction accepts a video directly:

```bash
python linguistic_features_per_second.py \
  --video /path/to/movie.mp4 \
  --output-dir ./outputs \
  --lmstudio-base-url http://127.0.0.1:1234 \
  --llm-model google/gemma-4-12b
```

To force exact full-video coverage, pass the number of output seconds:

```bash
python linguistic_features_per_second.py \
  --video /path/to/movie.mp4 \
  --duration-sec 918 \
  --output-dir ./outputs \
  --llm-model google/gemma-4-12b
```

The script produces two files:

```text
outputs/<video_name>/<video_name>_transcript_per_second.csv
outputs/<video_name>/<video_name>_linguistic_features_per_second.csv
```

The transcript is written before LLM analysis. If a strict LM Studio call
fails, rerunning the same command reuses the transcript and does not repeat ASR.
Use `--force-transcribe` only when the transcript needs to be replaced.

To prepare only the cached per-second transcript before starting LM Studio:

```bash
python linguistic_features_per_second.py \
  --video /path/to/movie.mp4 \
  --output-dir ./outputs \
  --transcribe-only
```

Whisper word timestamps are assigned uniquely by word midpoint. Each output row
uses the half-open interval `[second, second+1)`. The same transcript word is
not copied into every second crossed by a longer ASR segment.

The default ASR configuration is:

```text
model: small
language: en
device: cpu
compute type: int8
sample rate: 16000 Hz
```

These settings can be changed with `--asr-model`, `--language`, `--asr-device`,
`--asr-compute-type`, and `--sample-rate`.

## Event-Level Processing

Event-level processing uses two related tables:

1. An event timing table defines the event boundaries. See
   `example/hcp_movie2_event_timing.csv` for an HCP movie timing example.
2. An event-text table contains the transcript aggregated within those
   boundaries. This is the file passed to `--event-csv`; see
   `example/event_text_example.csv`. A blank table containing all event IDs and
   matching HCP boundaries is provided as
   `example/hcp_movie2_event_text_template.csv`.

The timing table is included to document how the events were defined. The
linguistic script does not read it directly. Before running the script, align
the transcript to each interval and create a CSV with these required columns:

```text
event,time_period,text
Event 0,00:00-00:26,"Example dialogue assigned to Event 0."
Event 1,00:26-00:35,"Example dialogue assigned to Event 1."
```

`time_period` uses the half-open interval `[start, end)`: an event beginning at
`00:26` includes transcript content at or after 26 seconds and before its end.
The `event` labels and boundaries must match the timing table.

The included HCP timing example contains Events 0-29 and ends at `13:31`. It
documents the event segmentation used for that example; it is not a statement
of the source video's full file duration.

```bash
python linguistic_features_events_only.py \
  --event-csv /path/to/movie_text_channel_sentiment.csv \
  --lmstudio-base-url http://127.0.0.1:1234 \
  --llm-model google/gemma-4-12b \
  --output-dir ./outputs
```

Output:

```text
outputs/<movie_name>/<movie_name>_linguistic_features_events.csv
```

`example/example_linguistic_features_events.csv` contains synthetic values for
demonstrating the output schema. It is not a research result and should not be
used for analysis.

`example/example_linguistic_features_per_second.csv` is also synthetic and
demonstrates the per-second output schema.

When the input filename ends in `_text_channel_sentiment.csv`, that suffix is
removed to obtain `<movie_name>`. Otherwise, the input filename stem is used.

## Strict LLM Error Handling

By default, any LM Studio connection, model, timeout, JSON, or response-format
error stops the run. This prevents an apparently successful CSV containing
invalid zero-valued LLM results.

To reproduce the older fallback behavior, explicitly add:

```bash
--allow-llm-errors
```

With this option, failed LLM rows receive `mean_surprisal=0`,
`predictability_score=0`, and the error message is recorded in `lm_error`.
Do not treat these fallback zeros as valid measurements.

## Output Columns

| Column | Definition |
| --- | --- |
| `window_type` | Always `event` in this entry script. |
| `window_id` | Event label copied from the input. |
| `time_period` | Event time interval copied from the input. |
| `text` | Event text used for all calculations. |
| `token_count` | Number of lowercase English alphabetic/apostrophe tokens. |
| `unique_token_count` | Number of unique tokens. |
| `ttr` | Type-token ratio: unique tokens divided by all tokens. |
| `mattr_25` | Moving-average type-token ratio using a 25-token window; for shorter text, equals TTR. |
| `mean_word_length` | Mean token length in characters. |
| `std_word_length` | Population standard deviation of token length. |
| `mean_zipf_frequency` | Mean English Zipf frequency from wordfreq. |
| `low_frequency_ratio` | Fraction of tokens with Zipf frequency below 3.0. |
| `very_low_frequency_ratio` | Fraction of tokens with Zipf frequency below 2.0. |
| `wordfreq_available` | `1` when wordfreq was imported, otherwise `0`. |
| `mean_sentence_length` | Mean number of alphabetic tokens per spaCy sentence. |
| `clause_density` | Number of selected dependency clause labels divided by sentence count. |
| `dep_tree_depth_mean` | Mean dependency-tree depth across alphabetic spaCy tokens. |
| `mean_surprisal` | LLM-estimated uncertainty on a custom 0-20 scale. |
| `predictability_score` | LLM-estimated predictability in `[0,1]`. |
| `llm_model` | Model identifier requested from LM Studio. |
| `lm_note` | Short explanation returned by the LLM. |
| `lm_error` | Error text when `--allow-llm-errors` is enabled and a call fails. |

Per-second output replaces `window_type`, `window_id`, and the event interval
with `second` and `time_period`. All linguistic feature definitions are
otherwise identical.

## Reproducibility and Limitations

- Use the same LM Studio version, model file, quantization, model identifier,
  prompt, and decoding settings for all stimuli.
- `--llm-model` is intentionally required. Changing the selected model changes
  the measurement procedure, so outputs from different models should not be
  pooled or compared without explicit validation.
- The script sets LLM temperature to `0`, but local inference can still vary by
  runtime, model format, and implementation.
- Short event text can make TTR, MATTR, syntax, and LLM estimates unstable.
- The tokenizer is English-specific and retains alphabetic words and internal
  apostrophes only.
- spaCy syntax features depend on `en_core_web_sm`. If it cannot be loaded, the
  script prints a warning and uses heuristic approximations.
- If wordfreq cannot be imported, frequency fields are zero and
  `wordfreq_available=0`.
- Empty text receives zero LLM scores with `lm_note=NO_TEXT` and is not sent to
  LM Studio.
- Per-second linguistic values depend on ASR accuracy and word-timestamp
  alignment. Errors in transcription or timing propagate into all downstream
  features.

For research use, record package versions:

```bash
python -m pip freeze > environment-lock.txt
```

## Data and Licensing

Do not commit copyrighted transcripts, identifiable participant data, private
API tokens, or restricted movie materials unless redistribution is permitted.
Review the licenses and usage terms for LM Studio, the selected model, spaCy,
and wordfreq before distribution or commercial use.
