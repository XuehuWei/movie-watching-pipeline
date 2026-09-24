# Movie-Watching Stimulus Feature Extraction Pipeline

This repository contains reproducible pipelines for extracting time-resolved
social-visual and speech-conversation features from movie stimuli.

Features can be extracted at:

- Per-second resolution
- Event-level resolution based on a segmentation CSV

## Modules

### Social-Visual Features

Directory:

[stimulus_feature_extraction/social_visual_features_release](stimulus_feature_extraction/social_visual_features_release)

This module uses YOLO, MediaPipe, OpenCV, and NumPy to extract:

- Person and face counts
- Person and face screen area
- Interpersonal distance and proximity
- Mutual-gaze proxy
- Facial-expression intensity
- Gesture intensity
- Heuristic facial-affect scores

### Speech and Conversation Features

Directory:

[stimulus_feature_extraction/speech_conversation_features_release](stimulus_feature_extraction/speech_conversation_features_release)

This module uses FFmpeg, pyannote.audio, and openSMILE to extract:

- Speech presence and speech ratio
- Speaker count
- Overlapping speech
- Speaker turn changes and turn-taking rate
- Prosody arousal
- Loudness
- Pitch variability
- Jitter

## Repository Structure

```text
movie-watching-pipeline/
├── README.md
└── stimulus_feature_extraction/
    ├──visual_social _features_release/
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── social_features_yolo_mediapipe.py
    │   ├── social_features_yolo_mediapipe_per_second.py
    │   └── social_features_yolo_mediapipe_events_only.py
    └── speech_conversation_features_release/
        ├── README.md
        ├── requirements.txt
        ├── install_opensmile.sh
        ├── speech_conversation_features.py
        ├── speech_conversation_features_per_second.py
        └── speech_conversation_features_events_only.py
