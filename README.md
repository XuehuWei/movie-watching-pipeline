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

The current public implementation includes the **stimulus feature extraction**
component, with pipelines for:

- Social-visual feature extraction
- Speech and conversation feature extraction
- Per-second feature extraction
- Event-level feature extraction

The dynamic ISC/ISFC, HMM-based brain-state, and statistical-modeling modules
will be added as they are finalized and validated.
