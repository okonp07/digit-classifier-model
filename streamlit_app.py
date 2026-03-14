"""Interactive app for spoken digit recognition."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from digit_recognition import DigitPredictor


st.set_page_config(page_title="Spoken Digit Recognition", page_icon="🎤", layout="wide")


def _plot_audio(audio: np.ndarray, sample_rate: int):
    import librosa

    figure, axes = plt.subplots(2, 1, figsize=(10, 6))
    times = np.arange(len(audio)) / sample_rate
    axes[0].plot(times, audio, color="#0f766e", linewidth=1)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13, n_fft=512, hop_length=256)
    image = axes[1].imshow(mfcc, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("MFCC Features")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Coefficient")
    figure.colorbar(image, ax=axes[1], shrink=0.8)
    figure.tight_layout()
    return figure


@st.cache_resource
def _load_predictor(model_name: str) -> DigitPredictor:
    return DigitPredictor(model_name)


def _predict(uploaded_file, predictor: DigitPredictor):
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    pred, conf, probs = predictor.predict_from_file(temp_path)
    audio = predictor.processor.load_audio(temp_path)
    report = predictor.processor.quality_report(audio)
    return temp_path, audio, pred, conf, probs, report


def main() -> None:
    st.title("Enhanced Spoken Digit Recognition")
    st.caption("Compare the baseline and enhanced checkpoints on your own audio files.")

    if "history" not in st.session_state:
        st.session_state.history = []
    if "audio_input_key" not in st.session_state:
        st.session_state.audio_input_key = 0
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    with st.sidebar:
        mode = st.radio(
            "Inference mode",
            ["Enhanced model", "Original model", "Compare both"],
            index=0,
        )
        st.markdown("Supported formats: WAV, MP3, M4A, FLAC, OGG")

    st.subheader("Choose input method")
    input_method = st.radio(
        "Audio source",
        ["Record with microphone", "Upload audio file"],
        horizontal=True,
    )

    audio_source = None
    if input_method == "Record with microphone":
        st.caption("Click record, allow microphone access in your browser, say one digit, then stop recording.")
        audio_source = st.audio_input(
            "Record a spoken digit",
            sample_rate=22050,
            key=f"audio-input-{st.session_state.audio_input_key}",
        )
    else:
        audio_source = st.file_uploader(
            "Upload an audio file containing one spoken digit",
            type=["wav", "mp3", "m4a", "flac", "ogg"],
            key=f"file-uploader-{st.session_state.file_uploader_key}",
        )

    controls_left, controls_right = st.columns([1, 3])
    with controls_left:
        if st.button("Clear current audio", use_container_width=True):
            if input_method == "Record with microphone":
                st.session_state.audio_input_key += 1
            else:
                st.session_state.file_uploader_key += 1
            st.rerun()

    if not audio_source:
        st.info("Record or upload audio to start.")
        return

    st.audio(audio_source)

    enhanced = _load_predictor("enhanced_digit_model.pth")
    original = _load_predictor("lightweight_digit_model.pth")

    if mode == "Compare both":
        left, right = st.columns(2)
        temp_path, audio, orig_pred, orig_conf, orig_probs, report = _predict(audio_source, original)
        _, _, enh_pred, enh_conf, enh_probs, _ = _predict(audio_source, enhanced)

        with left:
            st.subheader("Original model")
            st.metric("Prediction", orig_pred)
            st.metric("Confidence", f"{orig_conf:.1%}")
            st.bar_chart({str(idx): float(value) for idx, value in enumerate(orig_probs)})

        with right:
            st.subheader("Enhanced model")
            st.metric("Prediction", enh_pred)
            st.metric("Confidence", f"{enh_conf:.1%}")
            st.bar_chart({str(idx): float(value) for idx, value in enumerate(enh_probs)})

        selected_pred = enh_pred
        selected_conf = enh_conf
        selected_probs = enh_probs
    else:
        predictor = enhanced if mode == "Enhanced model" else original
        temp_path, audio, selected_pred, selected_conf, selected_probs, report = _predict(audio_source, predictor)
        st.subheader(mode)
        c1, c2 = st.columns(2)
        c1.metric("Prediction", selected_pred)
        c2.metric("Confidence", f"{selected_conf:.1%}")
        st.bar_chart({str(idx): float(value) for idx, value in enumerate(selected_probs)})

    st.pyplot(_plot_audio(audio, enhanced.processor.sample_rate))

    st.subheader("Audio quality checks")
    st.write(
        {
            "duration_seconds": round(report.duration_seconds, 3),
            "peak_amplitude": round(report.peak_amplitude, 4),
            "rms_amplitude": round(report.rms_amplitude, 4),
            "issues": list(report.issues) or ["No obvious issues detected."],
        }
    )

    st.session_state.history.append(
        {
            "file": audio_source.name,
            "prediction": selected_pred,
            "confidence": round(selected_conf, 4),
        }
    )
    st.subheader("Prediction history")
    st.dataframe(st.session_state.history, use_container_width=True)


if __name__ == "__main__":
    main()
