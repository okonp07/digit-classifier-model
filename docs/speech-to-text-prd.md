# Product Requirements Document

## Product

Speech-to-Text Transcription Update for `digit-classifier-model`

## Document purpose

This PRD defines the product goals, user experience, requirements, acceptance criteria, and release expectations for upgrading the existing audio-classification app into a general speech-to-text transcription experience.

## Background

The repository originally focused on spoken-digit recognition. That experience was constrained to a narrow classification problem and did not satisfy broader user needs for converting arbitrary spoken audio into written language. This update repurposes the app into a practical speech-to-text workflow that accepts microphone or uploaded audio and returns transcribed text with confidence signals and supporting diagnostics.

## Problem statement

Users can provide audio today, but the previous product only returns one of a fixed set of digit labels. That prevents the app from being useful for note capture, speech review, quick transcript generation, or general spoken-language prototyping.

We need a version of the app that:

- accepts free-form spoken input
- converts speech to written text
- exposes a confidence score users can interpret
- works from both microphone recordings and uploaded files
- preserves a simple, polished Streamlit experience

## Goals

- Convert spoken audio into readable text for arbitrary speech content.
- Support both browser microphone recording and uploaded audio files.
- Present an overall confidence score alongside the transcript.
- Show detected language and segment-level timing for transparency.
- Retain audio diagnostics to help users understand weak outputs.
- Keep the app deployable on Streamlit Community Cloud with a clear Python/runtime story.

## Non-goals

- Human-grade verbatim transcription in every acoustic condition.
- Real-time streaming transcription with partial token updates.
- Speaker diarization, translation, or subtitle export in this release.
- Fine-tuning or retraining the ASR model inside the app.
- Removal of all legacy digit-classification code from the repository.

## Target users

- Demo users exploring speech AI capabilities.
- Recruiters or reviewers evaluating the author’s applied AI product work.
- Developers who want a reusable local transcription component.
- Non-technical users who need quick text output from short recordings.

## Primary user stories

- As a user, I want to record speech directly in the browser and receive written text.
- As a user, I want to upload an audio file and see the transcript without extra setup.
- As a user, I want to know how confident the system is in the transcription.
- As a user, I want to understand whether audio quality may have affected the result.
- As a user, I want to review transcript segments with timing information.
- As a maintainer, I want a reusable Python module for transcription that the app can call cleanly.

## Experience summary

1. User opens the app.
2. User selects a transcription model size and optional language hint.
3. User records audio or uploads a supported file.
4. App processes the audio and runs speech-to-text transcription.
5. App displays:
   - transcript text
   - overall confidence score
   - detected language
   - language confidence when available
   - segment timeline
   - waveform and spectrogram
   - audio-quality checks
   - session history

## Functional requirements

### Input

- The app must accept browser microphone recordings.
- The app must accept uploaded `WAV`, `MP3`, `M4A`, `FLAC`, and `OGG` files.
- The app must allow the user to choose a transcription model size.
- The app must allow the user to provide a language hint or use auto-detection.

### Transcription output

- The app must return transcript text for arbitrary spoken content.
- The app must show an overall confidence score between `0` and `1`, presented as a percentage in the UI.
- The app must show detected language.
- The app must show language confidence when the ASR backend provides it.
- The app must show segment-level timing and confidence data when available.

### Diagnostics

- The app must display waveform and spectrogram visualizations.
- The app must surface audio-quality checks such as duration, RMS, clipping risk, and low-volume warnings.
- The app must preserve a session history of prior transcripts.

### Packaging and architecture

- The transcription logic must be available through a reusable Python API.
- Model loading should be lazy so the package stays importable even in partially provisioned environments.
- Optional dependencies should fail with clear runtime messages instead of opaque import errors.

## Non-functional requirements

- First-run model download behavior must be communicated clearly in the UI and docs.
- The app should remain usable on CPU-based environments.
- The UI must stay responsive for short-form recordings typical of demo use.
- The repository must include updated documentation for setup and usage.
- The repository must include tests for the new confidence and result-shaping logic.

## Proposed solution

- Use `faster-whisper` as the ASR backend for local transcription.
- Add a reusable `SpeechTranscriber` module that supports file and NumPy-array transcription.
- Derive a single confidence score from segment- and word-level probabilities when available.
- Update the Streamlit app to center the experience around transcript output instead of classification.
- Keep legacy digit-classification code for backward reference while routing the live app to speech-to-text.

## Success metrics

- User can obtain a transcript from both mic and file input in the same app session.
- Confidence score is displayed for every completed transcription.
- Detected language and segment timeline render successfully when data is available.
- Updated repo passes lint and test checks.
- Setup instructions are sufficient for a new user to run the app locally.

## Release acceptance criteria

- App accepts microphone and uploaded input without UI regressions.
- App displays transcript text and confidence score after successful transcription.
- App shows audio diagnostics and session history.
- README reflects the new speech-to-text product.
- A PRD ships in the repository describing the feature update.
- Verification passes:
  - `python3 -m py_compile streamlit_app.py evaluation.py training.py digit_recognition/*.py tests/*.py`
  - `flake8 digit_recognition tests training.py evaluation.py streamlit_app.py`
  - `pytest`

## Risks and mitigations

- Model download latency on first use:
  Mitigation: explain this in the sidebar and README.
- Lower transcription quality in noisy environments:
  Mitigation: provide quality checks and visual diagnostics.
- CPU-only deployments may be slower:
  Mitigation: expose smaller model sizes and default to CPU-safe behavior.
- Optional dependency failures:
  Mitigation: lazy-load ASR and audio-write dependencies with clear error messages.

## Rollout notes

- Ship as an in-place upgrade to the existing Streamlit app.
- Preserve the repository URL and deployment surface.
- Keep legacy files until a later cleanup release to avoid unnecessary breakage.

## Future enhancements

- Real-time streaming transcription.
- Downloadable transcript files.
- Translation mode.
- Speaker diarization.
- Editable transcript post-processing.
- Confidence highlighting at the word level in the UI.
