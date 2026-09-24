# Per-Second and Event-Level Low-Level Control Features

This module extracts low-level visual, auditory, and film-editing control
features from a movie. It writes a per-second CSV and can optionally aggregate
the same measurements into event intervals defined by a segmentation CSV.

No LLM, transcript, speech-recognition model, YOLO, MediaPipe, OpenFace,
pyannote, or Hugging Face token is used.

## Repository Files

```text
low_level_control_features.py
requirements.txt
.gitignore
README.md
example/MOVIE2_HO1_k29_event_segmentation.csv
example/example_low_level_control_features_per_second.csv
example/example_low_level_control_features_events.csv
example/Akeelah_and_the_Bee_trimmed/
  Akeelah_and_the_Bee_trimmed_low_level_control_features_per_second.csv
```

The Python script is standalone and does not import another script from this
repository.

## Features

| Feature | Definition |
| --- | --- |
| `motion_energy` | Mean absolute grayscale difference between consecutive frames, divided by 255. |
| `optical_flow_mean` | Mean Farneback dense-flow magnitude, normalized by the analyzed frame diagonal. |
| `brightness_mean` | Mean frame brightness extracted with pliers. |
| `contrast_std` | Mean grayscale pixel standard deviation divided by 255. |
| `saturation_mean` | Mean HSV saturation divided by 255. |
| `audio_rms` | One-second root-mean-square audio amplitude extracted with pliers. |
| `spectral_centroid_mean` | One-second spectral centroid in Hz, extracted with pliers/librosa. |
| `onset_strength_mean` | One-second spectral-flux onset-strength value, extracted with pliers/librosa. |
| `shot_cut_count` | Number of consecutive-frame histogram correlations below 0.70. |
| `visual_valid` | `1` when decoded video frames are available for the second, otherwise `0`. |

These are computational stimulus descriptors, not psychological labels.

## Software

| Software | Purpose |
| --- | --- |
| Python 3.10-3.12 recommended | Runs extraction and writes CSV files. |
| FFmpeg | Extracts mono 16 kHz PCM audio from the video. |
| OpenCV | Decodes frames and computes contrast, saturation, frame difference, optical flow, and shot cuts. |
| pliers 0.4.2 | Provides brightness, RMS, spectral-centroid, and onset-strength extractors. |
| librosa | Audio feature backend used by pliers. |
| NumPy and pandas | Numerical calculations and pliers result handling. |

## Installation

Install FFmpeg first. On macOS:

```bash
brew install ffmpeg
```

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

Create the Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
ffmpeg -version
python -c "import cv2, librosa, numpy, pandas, pliers; print('Installation OK')"
python low_level_control_features.py --help
```

## Per-Second Extraction

```bash
python low_level_control_features.py \
  --video /path/to/movie.mp4 \
  --duration-sec 918 \
  --output-dir ./outputs
```

Output:

```text
outputs/<video_stem>/<video_stem>_low_level_control_features_per_second.csv
```

`--duration-sec` is strongly recommended when the experimental duration is
known. For a duration of 918 seconds, the output contains exactly 918 rows,
numbered `second=0` through `second=917`. Without this argument, the script uses
the ceiling of the duration reported by the decoded video/audio streams, which
can include one additional partial second.

## Event-Level Extraction

Providing `--segments-csv` writes both per-second and event-level files:

```bash
python low_level_control_features.py \
  --video /path/to/7T_MOVIE2_HO1.mp4 \
  --segments-csv ./example/MOVIE2_HO1_k29_event_segmentation.csv \
  --duration-sec 918 \
  --output-dir ./outputs
```

Additional output:

```text
outputs/7T_MOVIE2_HO1/7T_MOVIE2_HO1_low_level_control_features_events.csv
```

The segmentation CSV must contain `Event`, `Start (mm:ss)`, and `End (mm:ss)`.
Additional columns are allowed and ignored. The included
[`MOVIE2_HO1_k29_event_segmentation.csv`](example/MOVIE2_HO1_k29_event_segmentation.csv)
demonstrates the accepted HCP movie event format.

Events use half-open intervals `[start, end)`. For example, `00:26-00:35`
includes per-second rows 26 through 34 but not row 35.

Event values are means of the included per-second rows, except
`shot_cut_count`, which is summed. `visual_valid_ratio` is the proportion of
included seconds with valid decoded video frames.

## Implementation Details

- Brightness is sampled at 1 Hz with pliers.
- Audio features use a hop length equal to the 16 kHz sample rate, producing
  one value per second.
- Dense optical flow is calculated on aspect-ratio-preserving frames with a
  maximum width of 320 pixels, then normalized by frame diagonal.
- Other visual measurements and cut detection use all decoded video frames.
- Output values are rounded to six decimal places.

## Example Outputs

The repository includes a real per-second output generated from the 140-second
example clip `Akeelah_and_the_Bee_trimmed.mp4`:

```text
example/Akeelah_and_the_Bee_trimmed/
  Akeelah_and_the_Bee_trimmed_low_level_control_features_per_second.csv
```

It contains exactly 140 rows (`second=0` through `second=139`). The source video
is not included because of file size and possible redistribution restrictions.

The two `example_low_level_control_features_*.csv` files contain small
synthetic values that demonstrate the per-second and event-level schemas. They
are not research results.

## Limitations

- Pixel difference and optical flow include actor motion, object motion,
  camera movement, zoom, and editing transitions.
- Histogram-based shot cuts are heuristic and can miss dissolves or count
  abrupt lighting changes.
- Spectral features summarize the complete soundtrack and therefore combine
  speech, music, and environmental sounds.
- Adjacent control variables can be correlated. Inspect correlations and use
  appropriate regularization, dimension reduction, or variable selection in
  statistical models.

## Reproducibility

Report the package versions, FFmpeg version, audio sample rate, explicit movie
duration, event segmentation source, and whether features were analyzed per
second or aggregated by event.
