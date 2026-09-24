# Stimulus Feature Extraction

This module extracts time-resolved stimulus features from naturalistic movie stimuli for downstream fMRI, eye-tracking, and brain-state analyses.

## Feature domains

- Visual features
- Social features
- Audio features
- Speech features
- Linguistic features
- Affective features

## Workflow

Movie stimuli
→ Feature extraction
→ Temporal alignment
→ Resampling
→ Quality control
→ Unified feature matrix

## Output

The output is a time × feature matrix aligned to the movie timeline for downstream analyses including:

- dynamic ISC
- dynamic ISFC
- HMM-based brain-state analysis
- eye-tracking analyses
- stimulus–brain modeling
