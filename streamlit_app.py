"""Interactive app for spoken digit recognition."""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from digit_recognition import DigitPredictor


st.set_page_config(page_title="Spoken Digit Recognition", page_icon="🎤", layout="wide")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
AUTHOR_IMAGE = ASSETS_DIR / "pic1.png"


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(242, 201, 76, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(15, 118, 110, 0.20), transparent 28%),
                linear-gradient(180deg, #f7f3ea 0%, #f4efe4 45%, #efe7d8 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 2rem 2rem 1.6rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #0f172a 0%, #134e4a 52%, #f59e0b 140%);
            color: #fff9ed;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.4rem;
        }
        .hero-shell:after {
            content: "";
            position: absolute;
            inset: auto -40px -55px auto;
            width: 220px;
            height: 220px;
            background: rgba(255, 248, 220, 0.12);
            border-radius: 999px;
            filter: blur(4px);
        }
        .hero-kicker {
            display: inline-block;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255, 248, 220, 0.14);
            border: 1px solid rgba(255, 248, 220, 0.22);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.7rem;
            line-height: 1.02;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0 0 0.65rem 0;
        }
        .hero-copy {
            max-width: 700px;
            font-size: 1.04rem;
            line-height: 1.6;
            color: rgba(255, 249, 237, 0.88);
            margin-bottom: 1.2rem;
        }
        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
        }
        .hero-pill {
            background: rgba(255, 249, 237, 0.12);
            border: 1px solid rgba(255, 249, 237, 0.2);
            border-radius: 999px;
            padding: 0.5rem 0.9rem;
            font-size: 0.9rem;
        }
        .section-card {
            background: rgba(255, 252, 245, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #102a26;
            margin-bottom: 0.25rem;
        }
        .section-copy {
            color: #4b5563;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }
        .mini-note {
            color: #5b5f68;
            font-size: 0.88rem;
            margin-top: 0.4rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 252, 245, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stMetricLabel"] {
            color: #5b5f68;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
        }
        .results-banner {
            margin: 0.25rem 0 0.8rem 0;
            padding: 0.85rem 1rem;
            border-radius: 18px;
            background: linear-gradient(90deg, rgba(15, 118, 110, 0.12), rgba(245, 158, 11, 0.12));
            border: 1px solid rgba(15, 118, 110, 0.12);
            color: #15312d;
            font-weight: 600;
        }
        .history-caption {
            color: #4b5563;
            font-size: 0.92rem;
            margin-top: -0.4rem;
            margin-bottom: 0.8rem;
        }
        .detail-card {
            background: rgba(255, 252, 245, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.35rem 1.35rem 1.1rem 1.35rem;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }
        .detail-card h3 {
            margin: 0 0 0.85rem 0;
            color: #102a26;
            font-size: 1.2rem;
            letter-spacing: -0.02em;
        }
        .detail-card p {
            color: #374151;
            line-height: 1.72;
            font-size: 0.98rem;
            margin-bottom: 0.8rem;
        }
        .detail-card ul {
            color: #374151;
            line-height: 1.72;
            font-size: 0.98rem;
            padding-left: 1.2rem;
            margin-top: 0.35rem;
        }
        .about-kicker {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.1);
            color: #0f766e;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.9rem;
        }
        .author-role {
            color: #0f766e;
            font-weight: 700;
            margin-top: -0.35rem;
            margin-bottom: 0.9rem;
        }
        .author-side-caption {
            text-align: center;
            color: #374151;
            line-height: 1.65;
            font-size: 0.98rem;
        }
        .author-side-caption strong {
            display: block;
            color: #102a26;
            font-size: 1.15rem;
            margin-bottom: 0.2rem;
        }
        .footer-shell {
            margin-top: 2.2rem;
            padding: 1.1rem 1rem 1.3rem 1rem;
            text-align: center;
            color: #4b5563;
            font-size: 0.93rem;
            line-height: 1.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(
    kicker: str = "Live Audio Demo",
    title: str = "Spoken Digit Recognition",
    copy: str = (
        "Record directly in the browser or upload a short clip, then compare the baseline and "
        "enhanced models with confidence scores, waveform previews, and MFCC visualizations."
    ),
    pills: list[str] | None = None,
) -> None:
    pills = pills or [
        "Microphone recording",
        "Model comparison",
        "Confidence tracking",
        "Audio quality checks",
    ]
    pills_html = "".join(f'<div class="hero-pill">{pill}</div>' for pill in pills)
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-copy">{copy}</div>
            <div class="hero-pills">
                {pills_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_intro(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _detail_card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="detail-card">
            <h3>{title}</h3>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        """
        <div class="footer-shell">
            <div>&copy; Okon Prince, 2026</div>
            <div>This is a simple mini project meant to illustrate the capacity to transform speach to text (STT)</div>
            <div>enquiries; okonp07@gmail.com, +234(0)9020000299</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _render_about_page() -> None:
    _render_hero(
        kicker="Project Overview",
        title="About",
        copy=(
            "This page explains what the spoken digit recognition system does, how the solution works end to end, "
            "and who built it."
        ),
        pills=["Project summary", "System workflow", "Model behavior", "Author profile"],
    )

    _detail_card(
        "About the project",
        """
        <p>
            This project is an interactive spoken-digit recognition system built to classify a short audio recording
            into one of the digits from <strong>0</strong> to <strong>9</strong>. It combines reusable machine learning
            code, packaged model checkpoints, and a Streamlit interface so the solution can be explored as a working app
            rather than only as a notebook experiment.
        </p>
        <p>
            The goal is to make the full workflow visible and usable: collect audio,
            preprocess it, transform it into machine-readable features, run it through
            the trained neural network, and present the prediction in a way that is
            understandable to a non-technical user.
        </p>
        """,
    )

    _detail_card(
        "How the solution works",
        """
        <p>
            The app accepts audio in two ways: a live browser microphone recording or
            an uploaded audio file. Once the sound is received, the system resamples it
            to a consistent sample rate, converts it to mono, trims away as much
            leading and trailing silence as possible, normalizes the loudness, and then
            selects the strongest active portion of the signal so the model focuses on
            the spoken digit rather than on silence or background noise.
        </p>
        <p>
            After preprocessing, the audio is converted into <strong>MFCC features</strong> (Mel-Frequency Cepstral
            Coefficients). These features are a compact representation of how the sound
            behaves across time and frequency, which makes them a practical input for
            speech-oriented models. The resulting feature map is passed into a
            lightweight convolutional neural network that was trained to output
            probabilities for the ten possible digits.
        </p>
        <p>
            During inference, the system can evaluate more than one closely related audio window and average the model
            outputs. This helps when the recorded speech is slightly early, late, or surrounded by silence. The final
            screen shows the predicted digit, the model confidence, a probability distribution across all classes, a
            waveform preview, MFCC visualization, and audio-quality checks that help explain poor predictions.
        </p>
        """,
    )

    _detail_card(
        "Why the app is structured this way",
        """
        <ul>
            <li>
                <strong>Usability:</strong> the app works for both quick browser testing
                and uploaded evaluation clips.
            </li>
            <li>
                <strong>Transparency:</strong> the prediction is supported by visual diagnostics
                instead of a raw number alone.
            </li>
            <li>
                <strong>Reusability:</strong> training, evaluation, and inference live in Python
                modules, not only in a notebook.
            </li>
            <li>
                <strong>Deployment readiness:</strong> model files, dependencies, and UI are
                packaged so the project can run on Streamlit Cloud.
            </li>
        </ul>
        """,
    )

    author_text_col, author_image_col = st.columns([1.45, 1], gap="large")
    with author_text_col:
        _detail_card(
            "About the Author",
            """
            <div class="about-kicker">Author Profile</div>
            <p><strong>Prince Okon</strong></p>
            <div class="author-role">
                Engineer &amp; Data Scientist
            </div>
            <p><strong>Senior Data Scientist at MIVA Open University</strong></p>
            <p>
                I design and deploy end-to-end data systems that turn raw data into production-ready intelligence.
            </p>
            <p>
                My core stack includes Python, Streamlit, BigQuery, Supabase, Hugging Face, PySpark, SQL,
                Machine Learning, LLMs, and Transformers.
            </p>
            <p>
                My work spans risk scoring systems, A/B testing, AI-powered dashboards,
                RAG pipelines, predictive analytics, and LLM-based solutions and AI research.
            </p>
            <p>
                Currently, I work as a Senior Data Scientist at MIVA Open University,
                building intelligent systems that drive analytics, decision support,
                and scalable AI innovation.
            </p>
            <p>
                <strong>I believe:</strong> models are trained, systems are engineered, impact is delivered.
            </p>
            """,
        )

    with author_image_col:
        _detail_card(
            "",
            """
            <div class="author-side-caption">
                <strong>Prince Okon</strong>
                <div>Engineer &amp; Data Scientist</div>
                <div>Senior Data Scientist at MIVA Open University</div>
            </div>
            """,
        )
        image_left, image_center, image_right = st.columns([0.12, 0.76, 0.12])
        with image_center:
            st.image(AUTHOR_IMAGE, use_container_width=True)


def _render_app_page() -> None:
    _render_hero()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "audio_input_key" not in st.session_state:
        st.session_state.audio_input_key = 0
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0

    with st.sidebar:
        st.markdown("### Control Panel")
        mode = st.radio(
            "Inference mode",
            ["Enhanced model", "Original model", "Compare both"],
            index=0,
        )
        st.markdown("Supported formats: WAV, MP3, M4A, FLAC, OGG")
        st.markdown("---")
        st.caption("Best results come from a single spoken digit in a short, clean recording.")

    _section_intro(
        "Input",
        (
            "Pick how you want to interact with the app. The microphone option works "
            "directly in the browser, while upload mode is handy for pre-recorded test clips."
        ),
    )

    input_method = st.radio(
        "Audio source",
        ["Record with microphone", "Upload audio file"],
        horizontal=True,
    )

    audio_source = None
    if input_method == "Record with microphone":
        st.caption(
            "Click record, allow microphone access in your browser, say one digit, then stop recording."
        )
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

    st.markdown(
        """
        <div class="results-banner">
            Prediction ready. Use the comparison mode to see how the original and enhanced
            checkpoints behave on the same clip.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    viz_col, qa_col = st.columns([1.5, 1])
    with viz_col:
        _section_intro(
            "Signal view",
            (
                "The waveform and MFCC plots help you see whether the recording is clear, "
                "clipped, or dominated by silence."
            ),
        )
        st.pyplot(_plot_audio(audio, enhanced.processor.sample_rate))

    with qa_col:
        _section_intro(
            "Quality checks",
            "These quick diagnostics flag common recording issues that can reduce prediction quality.",
        )
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
    _section_intro(
        "Prediction history",
        (
            "Track how the model responds across multiple takes so you can spot unstable "
            "recordings or compare different speaking styles."
        ),
    )
    st.markdown(
        '<div class="history-caption">Latest predictions from this session appear below.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)


def main() -> None:
    _inject_styles()

    with st.sidebar:
        page = st.radio("Page", ["App", "About"], index=0)

    if page == "About":
        _render_about_page()
    else:
        _render_app_page()

    _render_footer()


if __name__ == "__main__":
    main()
