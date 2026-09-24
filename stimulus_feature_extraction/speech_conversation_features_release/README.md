# Per-Second and Event-Level Speech and Conversation Features

This repository extracts per-second or event-level speech,
speaker-interaction, and acoustic prosody features from a video. It combines
FFmpeg, pyannote.audio, and openSMILE, then writes the results to CSV.

No LLM, LM Studio, ASR, transcript, or OpenFace model is used.

## Output Features

- Proportion of the event containing speech
- Number of speakers detected in the event
- Speaker turn-taking rate
- Proportion of overlapping speech
- Acoustic arousal proxy
- Mean loudness
- Pitch variability
- Mean jitter
- Per-second speech, speaker-count, overlap, turn-change, and prosody-valid flags

## Repository Files

Keep these files together:

```text
speech_conversation_features_events_only.py  # run this file
speech_conversation_features_per_second.py   # run for per-second output
speech_conversation_features.py              # shared functions; do not run first
install_opensmile.sh                         # one-time macOS/Linux installer
requirements.txt
.gitignore
README.md
example/MOVIE2_HO1_k29_event_segmentation.csv
example/example_speech_conversation_features.csv
example/example_speech_conversation_features_per_second.csv
```

`speech_conversation_features.py` is imported automatically by both entry
scripts. It does not need to be run before feature extraction.

## Processing Software

| Software | Required? | Purpose |
| --- | --- | --- |
| Python 3.10-3.12 | Yes | Runs the scripts and writes CSV output. |
| FFmpeg | Yes | Extracts mono, 16 kHz, PCM WAV audio from the input video. |
| pyannote.audio | Yes | Performs voice activity detection and speaker diarization. |
| PyTorch / torchaudio | Yes; installed with pyannote.audio | Executes the pyannote neural models and handles audio tensors. |
| Hugging Face account and read token | Yes | Downloads gated pyannote model files. |
| openSMILE `SMILExtract` | Yes | Extracts frame-level eGeMAPS low-level acoustic descriptors. |
| NumPy | Yes | Aggregates time intervals and calculates acoustic statistics. |

The pipeline has been tested locally with Python 3.13.5,
`pyannote.audio 4.0.4`, PyTorch 2.10.0, and openSMILE's eGeMAPS v01a
configuration. Python 3.10-3.12 is recommended for broader package-wheel
compatibility.

## How It Works

1. For event-level output, read `Event`, `Start (mm:ss)`, and `End (mm:ss)` from
   the segmentation CSV. For per-second output, construct `[0,1)`, `[1,2)`, and
   subsequent one-second intervals directly from the audio duration.
2. Use FFmpeg to convert the video's audio to mono 16 kHz PCM WAV.
3. Run `pyannote/speaker-diarization-3.1` over the complete audio track.
4. Run openSMILE `SMILExtract` with `eGeMAPSv01a.conf` over the complete audio.
5. Select diarization segments and acoustic frames inside each event or second
   using the half-open rule `[start, end)`.
6. Calculate speech, speaker, overlap, turn-taking, and prosody features.
7. Write an event-level or per-second CSV.

## Installation

### 1. Install FFmpeg

macOS with Homebrew:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

On Windows, install an FFmpeg distribution and add its `bin` directory to
`PATH`.

Verify:

```bash
ffmpeg -version
```

### 2. Create a Python environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The installation downloads PyTorch and pyannote dependencies and can require
several gigabytes of disk space.

### 3. Obtain pyannote model access

The default pipeline uses the gated Hugging Face model
`pyannote/speaker-diarization-3.1`.

1. Create or sign into a [Hugging Face account](https://huggingface.co/).
2. Accept the user conditions for
   [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0).
3. Accept the user conditions for
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1).
4. Create a [Hugging Face token](https://huggingface.co/settings/tokens) with
   read permission.
5. Either store it in the `HF_TOKEN` environment variable or pass it at runtime
   with `--hf-token`.

macOS/Linux:

```bash
export HF_TOKEN="your_hugging_face_read_token"
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="your_hugging_face_read_token"
```

Never commit a real token to GitHub, a shell script, README, notebook, or CSV.
If a token is exposed, revoke it in Hugging Face and create a replacement.

### 4. Install openSMILE

Recommended on macOS/Linux from the repository root:

```bash
bash install_opensmile.sh
```

The installer clones and builds openSMILE under:

```text
_third_party/opensmile/
```

Both Python entry scripts detect this repository-local installation
automatically. After this one-time installation, normal extraction commands do
not need `--opensmile-bin` or `--opensmile-config`.

The installer requires `git`, CMake, and a C++ compiler. On macOS, install
Xcode Command Line Tools and CMake if they are missing. On Linux, install the
platform's C++ build tools and CMake.

#### Manual or Windows installation

Download a platform release or build openSMILE from its
[official repository](https://github.com/audeering/opensmile). Detailed build
instructions are available in the
[openSMILE documentation](https://audeering.github.io/opensmile/).

```text
https://github.com/audeering/opensmile
```

For a source checkout on macOS/Linux:

```bash
git clone https://github.com/audeering/opensmile.git
cd opensmile
bash build.sh
```

The required files are normally located at:

```text
<opensmile>/build/progsrc/smilextract/SMILExtract
<opensmile>/config/egemaps/v01a/eGeMAPSv01a.conf
```

On Windows the executable is normally named `SMILExtract.exe`. Consult the
official openSMILE documentation for platform-specific prebuilt binaries or
build instructions.

If openSMILE is installed outside the repository, set the two paths:

```bash
export OPENSMILE_BIN="/absolute/path/to/opensmile/build/progsrc/smilextract/SMILExtract"
export OPENSMILE_CONFIG="/absolute/path/to/opensmile/config/egemaps/v01a/eGeMAPSv01a.conf"
```

Windows PowerShell:

```powershell
$env:OPENSMILE_BIN="C:\absolute\path\to\SMILExtract.exe"
$env:OPENSMILE_CONFIG="C:\absolute\path\to\eGeMAPSv01a.conf"
```

### 5. Verify the environment

```bash
python -c "import numpy, torch, torchaudio; from pyannote.audio import Pipeline; print('Python dependencies OK')"
./_third_party/opensmile/build/progsrc/smilextract/SMILExtract -h
python speech_conversation_features_events_only.py --help
python speech_conversation_features_per_second.py --help
```

## Event Segmentation CSV

Only event-level extraction needs this CSV. It must contain these exact column
names:

| Column | Meaning | Example |
| --- | --- | --- |
| `Event` | Event identifier copied to the output. | `Event 0` |
| `Start (mm:ss)` | Inclusive event start. `hh:mm:ss` is also accepted. | `00:00` |
| `End (mm:ss)` | Exclusive event end. `hh:mm:ss` is also accepted. | `00:26` |

The repository includes the complete segmentation example used for the HCP
movie event-level analysis:

[MOVIE2_HO1_k29_event_segmentation.csv](example/MOVIE2_HO1_k29_event_segmentation.csv)

Its format is:

```csv
Event,Start (TR),End (TR),Duration (TRs),Duration (s),Start (mm:ss),End (mm:ss)
Event 0,0,26,30,30,00:00,00:26
Event 1,26,35,9,9,00:26,00:35
```

The script reads only `Event`, `Start (mm:ss)`, and `End (mm:ss)`. The TR and
duration columns document the original annotation and are not used in feature
extraction. Other CSV columns are allowed and ignored. Rows with missing times
or `end <= start` are skipped.

## Run Event-Level Extraction

With environment variables configured:

```bash
python speech_conversation_features_events_only.py \
  --video /path/to/movie.mp4 \
  --segments-csv ./example/MOVIE2_HO1_k29_event_segmentation.csv \
  --output-dir ./outputs \
  --hf-token "<YOUR_HF_TOKEN>"
```

Alternatively, pass openSMILE paths explicitly:

```bash
python speech_conversation_features_events_only.py \
  --video /path/to/movie.mp4 \
  --segments-csv ./example/MOVIE2_HO1_k29_event_segmentation.csv \
  --output-dir ./outputs \
  --opensmile-bin /path/to/SMILExtract \
  --opensmile-config /path/to/eGeMAPSv01a.conf
```

CUDA is optional. On a compatible NVIDIA setup, add `--pyannote-cuda`.

The output is written to:

```text
outputs/<video_stem>/<video_stem>_speech_conversation_features.csv
```

`example/example_speech_conversation_features.csv` contains synthetic values
that demonstrate the output schema. It is not a research result and was not
calculated from a distributed video.

Temporary WAV and openSMILE frame files are automatically deleted after the
run.

## Run Per-Second Extraction

Per-second extraction does not require a segmentation CSV:

```bash
python speech_conversation_features_per_second.py \
  --video /path/to/movie.mp4 \
  --output-dir ./outputs \
  --hf-token "<YOUR_HF_TOKEN>"
```

To force an exact number of output seconds:

```bash
python speech_conversation_features_per_second.py \
  --video /path/to/movie.mp4 \
  --duration-sec 918 \
  --output-dir ./outputs \
  --hf-token "<YOUR_HF_TOKEN>"
```

The output is written to:

```text
outputs/<video_stem>/<video_stem>_speech_conversation_features_per_second.csv
```

With `--duration-sec 918`, the output contains 918 data rows numbered `0`
through `917`. Row `second=0` represents `[0,1)`, and row `second=917`
represents `[917,918)`.

When `--duration-sec` is omitted, the script reads the extracted WAV duration
and rounds it upward so that a final partial second is retained.

`example/example_speech_conversation_features_per_second.csv` contains
synthetic values showing the per-second schema.

## Output Definitions

### Event-Level Columns

| Column | Calculation and interpretation |
| --- | --- |
| `event` | Event label from the segmentation CSV. |
| `time_period` | Rounded event interval formatted as `mm:ss-mm:ss`. |
| `speech_ratio` | Union duration of all pyannote speech intervals divided by event duration. Range approximately `[0, 1]`. |
| `speaker_count` | Number of unique pyannote speaker labels overlapping the event. It is stored as a float but is integer-valued. |
| `turn_taking_rate` | Number of consecutive speaker-label changes divided by event duration in minutes. Units: changes/minute. |
| `overlap_speech` | Duration with at least two simultaneously active speaker segments divided by event duration. Range approximately `[0, 1]`. |
| `prosody_arousal` | Custom movie-normalized acoustic activation proxy in `[-1, 1]`; it is not an openSMILE standard variable or a clinical arousal rating. |
| `loudness_mean` | Mean openSMILE loudness value across frames in the event. Units follow the selected openSMILE descriptor/configuration. |
| `pitch_std` | Standard deviation of positive F0 values in the event. Units follow the F0 column emitted by the selected config. |
| `jitter_mean` | Mean local jitter across available event frames. |
| `diarization_mode` | Records the diarization backend (`pyannote`). |
| `prosody_mode` | Records the acoustic backend (`opensmile`). |

### Per-Second Columns

| Column | Calculation and interpretation |
| --- | --- |
| `second` | Zero-based second index. |
| `timestamp` | Start of the second formatted as `mm:ss`. |
| `speech` | `1` when any pyannote speech segment overlaps the second; otherwise `0`. |
| `speaker_count_second` | Number of unique pyannote speaker labels overlapping `[second, second+1)`. |
| `overlap_speech_second` | Fraction of the second during which at least two speaker segments are active. Range approximately `[0,1]`. |
| `turn_change_second` | `1` when a detected speaker-label transition starts within the second; otherwise `0`. Multiple changes in one second are still represented as `1`. |
| `prosody_valid` | `1` when speech is present and at least one acoustic validity condition is met (`loudness_mean > 0.01`, `pitch_std > 0`, or `jitter_mean > 0`); otherwise `0`. |
| `prosody_arousal` | Same movie-normalized acoustic activation proxy used in event-level output. |
| `loudness_mean` | Mean openSMILE loudness value across all acoustic frames in the second. |
| `pitch_std` | Standard deviation of positive F0 values in the second. |
| `jitter_mean` | Mean local jitter across available frames in the second. |
| `diarization_mode` | Records the diarization backend (`pyannote`). |
| `prosody_mode` | Records the acoustic backend (`opensmile`). |

`prosody_valid` is a quality flag only. The script does not replace invalid
prosody values with missing values or zero. Downstream analyses should filter
or mask prosody columns using `prosody_valid` when appropriate.

## Prosody Arousal Formula

Each openSMILE frame is standardized relative to the complete movie audio.
For event `e`:

```text
raw_e = 0.45 * mean(z(loudness)_e)
      + 0.30 * mean(abs(z(F0)_e))
      + 0.15 * mean(abs(z(jitter)_e))
      + 0.10 * mean(abs(z(shimmer)_e))

prosody_arousal_e = tanh(raw_e)
```

Therefore, `0` means approximately movie-average activation, positive values
mean higher activation relative to that movie, and negative values mean lower
activation. Values should not be compared across movies without additional
normalization and validation.

## Important Prosody Limitation

The current implementation runs openSMILE on the complete audio track and
aggregates every acoustic frame inside an event or second. It does **not** mask
openSMILE frames using pyannote speech segments. Background music, sound
effects, silence, and environmental noise can therefore affect
`prosody_arousal`, `loudness_mean`, `pitch_std`, and `jitter_mean`.

The pyannote output is used for `speech_ratio`, `speaker_count`,
`turn_taking_rate`, and `overlap_speech`, but not for filtering prosody frames.

## Additional Limitations

- Diarization labels are anonymous and local to one video; `SPEAKER_00` does not
  identify a real person across videos.
- Music, crowd noise, overlapping voices, short events, and poor audio quality
  can reduce diarization accuracy.
- `speaker_count` is the number of detected speaker labels, not a verified
  ground-truth cast count.
- Turn changes are inferred from sorted diarization segments and may be affected
  by overlap or segmentation errors.
- Event boundaries use `[start, end)`: start is included and end is excluded.
- The output is a computational stimulus description, not a clinical measure.

## Troubleshooting

### `ffmpeg was not found`

Install FFmpeg and confirm `ffmpeg -version` works in the same terminal.

### Hugging Face `401`, `403`, or gated-repository error

Confirm that `HF_TOKEN` is valid and that the account associated with it has
accepted the required pyannote model conditions. Some pyannote 4.x combinations
may also request access to `pyannote/speaker-diarization-community-1`; follow the
repository named in the error and accept its conditions if appropriate.

### `openSMILE config not found`

Use an absolute path to `eGeMAPSv01a.conf` and verify `OPENSMILE_CONFIG` in the
same terminal used to run Python.

### `openSMILE executable not found`

Set `OPENSMILE_BIN` to the compiled or downloaded `SMILExtract` executable. On
macOS/Linux, ensure it is executable with `chmod +x /path/to/SMILExtract`.

### CPU execution is slow

Speaker diarization is computationally expensive. Use `--pyannote-cuda` only
when PyTorch reports that CUDA is available. Apple Metal/MPS is not selected by
this script.

## Reproducibility

For research use, report at least:

- pyannote pipeline name and package version
- openSMILE version and configuration file
- FFmpeg version
- audio sample rate
- event segmentation source and time convention
- CPU/GPU execution mode
- the custom prosody-arousal formula shown above

Record installed Python packages with:

```bash
python -m pip freeze > environment-lock.txt
```

Do not commit `environment-lock.txt` without reviewing it for local paths or
private package sources.

## Licenses and Data Privacy

Add a license for this repository before public distribution. FFmpeg,
pyannote.audio, pretrained Hugging Face models, PyTorch, and openSMILE have
their own licenses or model terms. Review each project's terms for the intended
academic or commercial use.

Audio and diarization results can contain sensitive information. Do not upload
copyrighted videos, extracted WAV files, participant recordings, or private
tokens unless redistribution and data-sharing permissions allow it.
