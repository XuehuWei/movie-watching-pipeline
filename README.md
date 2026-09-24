# Movie-Watching Stimulus Feature Extraction and fMRI Analysis Pipeline

A reproducible analysis pipeline for naturalistic movie-fMRI, including
stimulus feature extraction, dynamic inter-subject correlation and functional
connectivity analysis (ISC/ISFC), hidden Markov model (HMM)-based brain-state
analysis, and statistical modeling.

## Pipeline Scope

The complete pipeline is designed to include four major components:

1. **Stimulus feature extraction**
   - Social-visual features
   - Speech and conversation features
   - Linguistic and semantic features
   - Low-level visual and auditory control features
   - Per-second and event-level representations

2. **Dynamic ISC/ISFC analysis**
   - Time-resolved inter-subject correlation (ISC)
   - Inter-subject functional correlation (ISFC)
   - Sliding-window and event-level neural synchrony
   - Group and condition comparisons

3. **HMM-based brain-state analysis**
   - Identification of recurring brain states
   - State-transition probabilities
   - State occupancy and dwell time
   - Alignment between brain-state transitions and movie events

4. **Statistical modeling**
   - Relationships between stimulus features and neural responses
   - Event-level and time-resolved regression models
   - Mixed-effects and group-level analyses
   - Control for low-level sensory and temporal confounds

## Current Implementation Status


The current public implementation includes the following stimulus feature
extraction pipelines:

- Social-visual feature extraction at per-second and event-level resolutions
- Speech and conversation feature extraction at per-second and event-level resolutions
- Linguistic feature extraction at per-second and event-level resolutions

The dynamic ISC/ISFC, HMM-based brain-state, semantic-feature, low-level
sensory-control, and statistical-modeling modules will be added as they are
finalized and validated.

## Available Modules

### Social-Visual Features

[stimulus_feature_extraction/social_visual_features_release](stimulus_feature_extraction/visual_social_feature_release)

This module extracts person, face, proximity, gaze, expression, gesture, and
facial-affect features using YOLO, MediaPipe, OpenCV, and NumPy.

Available resolutions:

- Per-second
- Event-level

### Speech and Conversation Features

[stimulus_feature_extraction/speech_conversation_features_release](stimulus_feature_extraction/speech_conversation_features_release)

This module extracts speech presence, speaker count, overlapping speech,
turn-taking, loudness, pitch variability, jitter, and prosody-arousal features
using FFmpeg, pyannote.audio, and openSMILE.

Available resolutions:

- Per-second
- Event-level

### Linguistic Features

[stimulus_feature_extraction/linguistic_features_release](stimulus_feature_extraction/linguistic_features_release)

This module extracts lexical diversity, word-frequency, word-length, syntactic
complexity, and LLM-estimated language uncertainty features using
faster-whisper, spaCy, wordfreq, and a user-selected LM Studio model.

Available resolutions:

- Per-second directly from video
- Event-level from a prepared event-text CSV

## Data and Security

This repository does not include copyrighted movie files, extracted audio,
restricted transcripts, private participant data, model weights, or access
tokens.

Users are responsible for ensuring that they have permission to process and
share input videos, transcripts, neuroimaging data, and derived results.

## Research Use

The extracted variables are computational stimulus descriptors. Heuristic gaze,
facial-expression, emotion, prosody, and LLM-estimated language variables should
not be interpreted as clinical or diagnostic measurements.
