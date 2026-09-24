# Per-Second and Event-Level Social-Visual Feature Extraction

This repository extracts social-visual features at either one-second or
event-defined resolution. It combines YOLO person detection with MediaPipe Face
Mesh (or an OpenCV face-detector fallback) and writes the aggregated results to
CSV.

The pipeline does **not** use an LLM. Per-second extraction does not require
annotations; event-level extraction requires a segmentation CSV.

## Features

- Person count, presence, and normalized screen area
- Face count, presence, and normalized screen area
- Interpersonal proximity
- Frontal-face / mutual-gaze proxy
- Facial-expression intensity proxy
- Body-motion / gesture-intensity proxy
- Heuristic facial-affect scores: happy, sad, angry, calm, and fearful

## Repository Files

Keep all of these files in the same directory:

```text
social_features_yolo_mediapipe_per_second.py  # command-line entry point
social_features_yolo_mediapipe_events_only.py # event-level command-line entry point
social_features_yolo_mediapipe.py             # detection and aggregation functions
requirements.txt                              # Python dependencies
README.md
example/MOVIE2_HO1_k29_event_segmentation.csv # event segmentation example
```

`social_features_yolo_mediapipe.py` is a shared library module. **Do not run it
first or execute it directly.** The per-second and event-level entry scripts
import its detection and aggregation functions automatically. To run the
pipeline, execute one of these two entry points:

```text
social_features_yolo_mediapipe_per_second.py
social_features_yolo_mediapipe_events_only.py
```

## Software Used

No commercial desktop application is required. The pipeline is a Python
program built from the following components:

| Software | Required? | Role in this pipeline |
| --- | --- | --- |
| Python | Yes | Runs the scripts, command-line interface, and CSV input/output. Python 3.10-3.12 is recommended; newer versions depend on wheel availability for the platform. |
| Ultralytics YOLO | Yes | Loads `yolov8n.pt` and detects people in each sampled frame. Only YOLO class `0` (`person`) is retained. |
| PyTorch | Yes, installed with Ultralytics | Executes the YOLO neural network. A GPU is not required; CPU execution is supported but slower. |
| OpenCV | Yes | Opens and decodes the video, reads frames, obtains FPS/duration, and provides the Haar face-detector fallback. |
| MediaPipe | Recommended | Detects Face Mesh landmarks used for face area, frontalness, facial-expression intensity, and facial-affect proxies. |
| NumPy | Yes | Calculates distances, means, percentiles, normalized areas, and one-second/event aggregates. |

The following software is **not** used by these scripts:

- LM Studio or any LLM
- OpenFace
- pyannote.audio
- openSMILE
- Hugging Face tokens
- the `ffmpeg` command-line program

OpenCV may use its bundled video-codec libraries internally, but users do not
need to call `ffmpeg` separately.

## Model Used

The default person detector is Ultralytics YOLOv8 Nano:

```text
yolov8n.pt
```

The weights are not included in this repository. If `yolov8n.pt` is not already
available locally, Ultralytics normally downloads it on the first run. This
requires an internet connection once. For offline use, download the weights in
advance and pass the local path with `--yolo-model`.

MediaPipe Face Mesh is included in the MediaPipe Python package and does not
require a separate model argument in this implementation.

## System Requirements

- macOS, Linux, or Windows
- Python 3.10-3.12 recommended
- Enough free disk space for the Python environment and YOLO weights
- A local video readable by OpenCV, preferably MP4/H.264
- Internet access during installation and the first YOLO run
- GPU optional; CPU-only execution is supported

Runtime depends mainly on video duration, resolution, hardware, and
`--sample-fps`. A larger sampling rate analyzes more frames and takes longer.

## Installation

### 1. Download the repository

All three Python files must remain in the same directory. Do not run
`social_features_yolo_mediapipe.py` directly; it is imported by the two entry
scripts.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the recommended dependencies

```bash
python -m pip install -r requirements.txt
```

The `ultralytics` and `mediapipe` packages install their required PyTorch and
OpenCV dependencies automatically. Avoid manually installing several OpenCV
variants (`opencv-python`, `opencv-python-headless`, and
`opencv-contrib-python`) into the same environment unless necessary, because
they share the same `cv2` import name.

### 4. Verify the installation

```bash
python -c "import cv2, numpy, mediapipe; from ultralytics import YOLO; print('Installation OK')"
python social_features_yolo_mediapipe_per_second.py --help
python social_features_yolo_mediapipe_events_only.py --help
```

### Installation without MediaPipe

If MediaPipe has no compatible wheel for the operating system or Python
version, create a clean environment and install only the base stack:

```bash
python -m pip install numpy ultralytics opencv-python
```

The scripts then print a warning and use OpenCV Haar face detection. Person
detection remains available, but face, expression, and gaze proxies are less
precise than the MediaPipe path.

## Per-Second Extraction

```bash
python social_features_yolo_mediapipe_per_second.py \
  --video /path/to/movie.mp4 \
  --output-dir ./outputs \
  --sample-fps 5
```

To force an exact output duration, pass an integer number of seconds:

```bash
python social_features_yolo_mediapipe_per_second.py \
  --video /path/to/movie.mp4 \
  --duration-sec 918 \
  --output-dir ./outputs \
  --sample-fps 5
```

The result is written to:

```text
outputs/<video_stem>/<video_stem>_social_features_per_second.csv
```

For example, `movie.mp4` produces:

```text
outputs/movie/movie_social_features_per_second.csv
```

### Included Example Output

The repository includes a per-second CSV generated from the 140-second example
clip `Akeelah_and_the_Bee_trimmed.mp4` at `--sample-fps 5`:

```text
example/Akeelah_and_the_Bee_trimmed/
  Akeelah_and_the_Bee_trimmed_social_features_per_second.csv
```

The CSV contains 140 data rows (`second=0` through `second=139`). The source
video is not included because of its size and possible redistribution rights.

## Event-Level Extraction

Event-level extraction requires a segmentation CSV in addition to the video:

```bash
python social_features_yolo_mediapipe_events_only.py \
  --video /path/to/movie.mp4 \
  --segments-csv ./example/MOVIE2_HO1_k29_event_segmentation.csv \
  --output-dir ./outputs \
  --sample-fps 5
```

The result is written to:

```text
outputs/<video_stem>/<video_stem>_social_features_events.csv
```

The segmentation CSV must contain these columns:

| Required column | Description | Example |
| --- | --- | --- |
| `Event` | Event identifier copied to output `window_id`. | `Event 0` |
| `Start (mm:ss)` | Event start time. `hh:mm:ss` is also accepted. | `00:00` |
| `End (mm:ss)` | Event end time. `hh:mm:ss` is also accepted. | `00:26` |

The repository includes a complete event segmentation example:

[MOVIE2_HO1_k29_event_segmentation.csv](example/MOVIE2_HO1_k29_event_segmentation.csv)

Its first rows follow this structure:

```csv
Event,Start (TR),End (TR),Duration (TRs),Duration (s),Start (mm:ss),End (mm:ss)
Event 0,0,26,30,30,00:00,00:26
Event 1,26,35,9,9,00:26,00:35
```

The event-level script reads only `Event`, `Start (mm:ss)`, and `End (mm:ss)`.
The TR and duration columns are retained to document the original annotation
but are not used in feature extraction. Additional columns are allowed and
ignored. Event rows with missing times or `end <= start` are skipped.

## Time Convention

Each CSV row represents a half-open one-second interval:

```text
second 0 -> [0.0, 1.0)
second 1 -> [1.0, 2.0)
second 2 -> [2.0, 3.0)
```

The start time is included and the end time is excluded. With
`--duration-sec 918`, the output contains 918 rows, numbered `0` through `917`.

If `--duration-sec` is omitted, duration is calculated from the video's frame
count and native frame rate and then rounded to the nearest whole second.

Event-level intervals use the same half-open convention. For example, an event
from `00:26` to `00:35` includes sampled frames at times `t` satisfying
`26 <= t < 35`.

## Command-Line Options

| Option | Default | Description |
| --- | ---: | --- |
| `--video` | required | Input video path. |
| `--output-dir` | `outputs/dynamic_social_4second_noalltime` | Root output directory. |
| `--yolo-model` | `yolov8n.pt` | Ultralytics YOLO weights path or model name. |
| `--sample-fps` | `5.0` | Number of video frames analyzed per second. |
| `--yolo-conf` | `0.25` | YOLO detection-confidence threshold. |
| `--close-threshold` | `0.15` | Maximum normalized center distance used to count a close person pair. |
| `--duration-sec` | detected duration | Exact number of one-second output rows. |

`--segments-csv` is required by the event-level script and is not used by the
per-second script. `--duration-sec` is available only in the per-second script.

Increasing `--sample-fps` generally improves temporal coverage but increases
runtime. The default analyzes approximately five frames within each second.

## Output Columns

| Column | Definition |
| --- | --- |
| `second` | Zero-based second index. |
| `window_id` | Event identifier. Present only in event-level output. |
| `time_period` | Human-readable interval, such as `00:00-00:01`. |
| `start_s`, `end_s` | Numeric boundaries of the half-open interval `[start_s, end_s)`. |
| `people_count` | Mean number of YOLO class-0 person detections across sampled frames. |
| `people_count_max` | Maximum detected person count in the second. |
| `people_count_p90` | 90th percentile of detected person counts in the second. |
| `people_presence_ratio` | Fraction of sampled frames containing at least one detected person. |
| `person_area_mean` | Mean detected-person bounding-box area divided by full-frame area. |
| `people_close_count` | Mean number of person pairs whose normalized center distance is at or below `--close-threshold`. |
| `face_count` | Mean number of detected faces across sampled frames. |
| `face_count_max` | Maximum detected face count in the second. |
| `face_count_p90` | 90th percentile of detected face counts in the second. |
| `face_presence_ratio` | Fraction of sampled frames containing at least one detected face. |
| `face_area_mean` | Mean face bounding-box area divided by full-frame area. |
| `interpersonal_distance_mean` | Mean pairwise person-center distance normalized by the frame diagonal. |
| `mutual_gaze_score` | `[0, 1]` proxy based on face frontalness and distance between detected faces. It is not eye tracking. |
| `facial_expression_intensity` | `[0, 1]` heuristic based on mouth opening, eyebrow geometry, and eye opening. |
| `gesture_intensity` | Mean nearest-matched person-center displacement between sampled frames, normalized by frame diagonal. |
| `happy`, `sad`, `angry`, `calm`, `fearful` | Normalized heuristic facial-affect scores. The five values sum to approximately 1 for each output row. |

Mean counts such as `people_count` and `face_count` can contain decimals. This is
expected: they are averages across sampled frames, not a literal count from one
frame. Use the corresponding `_max` column when an integer-like peak count is
needed.

## Method Summary

1. Open the video with OpenCV and obtain its native frame rate and duration.
2. Sample frames at the requested `--sample-fps`.
3. Detect people with YOLO and retain only class `0` (`person`).
4. Detect facial landmarks with MediaPipe Face Mesh. If unavailable, use the
   OpenCV frontal-face Haar cascade.
5. Calculate frame-level spatial, gaze, motion, and facial-expression proxies.
6. Aggregate sampled frames into one-second intervals or CSV-defined event
   intervals using means, maxima, percentiles, and presence ratios.
7. Write one CSV row for every second or event.

## Interpretation and Limitations

- The facial-affect columns are geometry-based heuristics, not predictions from
  a validated facial-emotion recognition model.
- `mutual_gaze_score` estimates a social-visual configuration from frontalness
  and proximity. It does not measure actual gaze direction or eye contact.
- `gesture_intensity` measures detected-person displacement and can include
  camera motion, cuts, tracking mismatch, or ordinary movement.
- Occlusion, profile faces, small faces, unusual camera angles, and rapid cuts
  can reduce detection accuracy.
- When no face is detected, the implementation assigns `calm=1` and the other
  affect scores `0`. Use `face_presence_ratio` to identify and optionally exclude
  these seconds from facial-affect analyses.
- Outputs should be treated as computational stimulus descriptors, not clinical
  or diagnostic measurements.

For reproducible research, report the YOLO weights, package versions, sampling
rate, confidence threshold, close-pair threshold, and whether MediaPipe or the
OpenCV fallback was used.

## Troubleshooting

### `ultralytics is not installed`

Activate the intended environment and reinstall the requirements:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### MediaPipe warning

The script can still run with the OpenCV fallback. For more reliable facial
landmarks, install a MediaPipe build compatible with your Python version and
platform.

### YOLO weights cannot be downloaded

Download compatible Ultralytics weights separately and provide their path:

```bash
python social_features_yolo_mediapipe_per_second.py \
  --video /path/to/movie.mp4 \
  --yolo-model /path/to/yolov8n.pt
```

### Video cannot be opened

Confirm that the path is correct and that the OpenCV build supports the video's
codec. Re-encoding the video to H.264 MP4 often resolves codec incompatibility.

## License and Model Terms

Add a project license before public distribution. The code, Ultralytics models,
and MediaPipe dependencies may have separate license or usage terms; review the
terms that apply to your intended use before redistribution.
