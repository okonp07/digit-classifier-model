"""Interactive app for speech-to-text transcription."""

from __future__ import annotations

import tempfile
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from digit_recognition.transcriber import SpeechTranscriber, TranscriptionResult


st.set_page_config(page_title="Speech-to-Text Transcription", page_icon="🎤", layout="wide")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
AUTHOR_IMAGE = ASSETS_DIR / "pic1.png"
REPO_URL = "https://github.com/okonp07/digit-classifier-model"
FUTURE_DEVELOPMENT_URL = f"{REPO_URL}/blob/main/future-development.md"
HeroPill = tuple[str, str]


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
        a.hero-pill {
            display: block;
            background: rgba(255, 249, 237, 0.12);
            border: 1px solid rgba(255, 249, 237, 0.2);
            border-radius: 999px;
            padding: 0.5rem 0.9rem;
            font-size: 0.9rem;
            color: #fff9ed;
            text-decoration: none;
            transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
        }
        a.hero-pill:hover {
            transform: translateY(-1px);
            background: rgba(255, 249, 237, 0.18);
            border-color: rgba(255, 249, 237, 0.32);
        }
        a.hero-pill:focus,
        a.hero-pill:focus-visible {
            outline: 2px solid rgba(255, 249, 237, 0.8);
            outline-offset: 2px;
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
        .section-card,
        .detail-card,
        .results-banner {
            scroll-margin-top: 1.2rem;
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
        .transcript-shell {
            background: rgba(255, 252, 245, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.06);
            min-height: 220px;
        }
        .transcript-label {
            color: #0f766e;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.8rem;
            font-weight: 700;
        }
        .transcript-body {
            color: #102a26;
            font-size: 1.08rem;
            line-height: 1.85;
            white-space: pre-wrap;
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
    kicker: str = "Live Speech-to-Text",
    title: str = "Speech-to-Text Transcription",
    copy: str = (
        "Record directly in the browser or upload a spoken clip, then turn it into written text "
        "with a confidence score, detected language, and audio diagnostics."
    ),
    pills: list[HeroPill] | None = None,
) -> None:
    pills = pills or [
        ("Microphone recording", "#input-section"),
        ("Transcript output", "#results-section"),
        ("Confidence score", "#results-section"),
        ("Audio quality checks", "#quality-checks-section"),
    ]
    pills_html = "".join(
        f'<a class="hero-pill" href="{escape(href, quote=True)}">{escape(label)}</a>'
        for label, href in pills
    )
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">{escape(kicker)}</div>
            <div class="hero-title">{escape(title)}</div>
            <div class="hero-copy">{escape(copy)}</div>
            <div class="hero-pills">
                {pills_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_intro(title: str, copy: str, anchor_id: str | None = None) -> None:
    anchor_attr = f' id="{escape(anchor_id, quote=True)}"' if anchor_id else ""
    st.markdown(
        f"""
        <div class="section-card"{anchor_attr}>
            <div class="section-title">{escape(title)}</div>
            <div class="section-copy">{escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _detail_card(
    title: str,
    body_html: str,
    kicker: str | None = None,
    anchor_id: str | None = None,
) -> None:
    anchor_attr = f' id="{escape(anchor_id, quote=True)}"' if anchor_id else ""
    title_html = f"<h3>{escape(title)}</h3>" if title else ""
    kicker_html = f'<div class="about-kicker">{escape(kicker)}</div>' if kicker else ""
    card_html = f'<div class="detail-card"{anchor_attr}>{title_html}{kicker_html}{body_html}</div>'
    st.markdown(card_html, unsafe_allow_html=True)


def _render_footer() -> None:
    st.markdown(
        """
        <div class="footer-shell">
            <div>&copy; Okon Prince, 2026</div>
            <div>
                This project is meant for Educational and research purposes only.
                It is not to be used for commercial purposes.
            </div>
            <div>enquiries; okonp07@gmail.com</div>
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

    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=64, n_fft=1024, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    image = axes[1].imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("Mel Spectrogram")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel Bin")
    figure.colorbar(image, ax=axes[1], shrink=0.8)
    figure.tight_layout()
    return figure


def _html_paragraphs(paragraphs: list[str]) -> str:
    return "\n".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs)


def _html_bullets(items: list[tuple[str, str]]) -> str:
    bullet_html = "".join(
        f"<li><strong>{escape(label)}:</strong> {escape(description)}</li>"
        for label, description in items
    )
    return f"<ul>{bullet_html}</ul>"


def _author_profile_html() -> str:
    paragraphs = [
        (
            "Okon Prince\n"
            "Senior Data Scientist at MIVA Open University | AI Engineer & Data Scientist"
        ),
        "I design and deploy end-to-end data systems that turn raw data into production-ready intelligence.",
        (
            "My core stack includes Python, Streamlit, BigQuery, Deep Learning, Supabase, "
            "Hugging Face, PySpark, SQL, Machine Learning, LLMs, and Transformers."
        ),
        (
            "My work spans risk scoring systems, A/B testing, Traditional and AI-powered "
            "dashboards, RAG pipelines, predictive analytics, Image processing and analytics, "
            "LLM-based solutions and AI research."
        ),
        (
            "Currently, I work as a Senior Data Scientist in the department of Research and "
            "Development at MIVA Open University, where I carry out AI / ML Research and "
            "build intelligent systems that drive analytics, decision support and scalable AI innovation."
        ),
    ]
    return "\n".join(
        [
            _html_paragraphs(paragraphs),
            (
                f"<p><strong>{escape('I believe:')}</strong> "
                f"{escape('models are trained, systems are engineered and impact is delivered.')}</p>"
            ),
        ]
    )


def _render_sidebar_navigation() -> str:
    if "page" not in st.session_state:
        st.session_state.page = "App"

    with st.sidebar:
        nav_app, nav_about = st.columns(2)
        if nav_app.button(
            "App",
            use_container_width=True,
            type="primary" if st.session_state.page == "App" else "secondary",
        ):
            st.session_state.page = "App"
            st.rerun()
        if nav_about.button(
            "About",
            use_container_width=True,
            type="primary" if st.session_state.page == "About" else "secondary",
        ):
            st.session_state.page = "About"
            st.rerun()
        st.link_button("future development", FUTURE_DEVELOPMENT_URL, use_container_width=True)
        st.markdown("---")

    return st.session_state.page


@st.cache_resource
def _load_transcriber(model_size: str) -> "SpeechTranscriber":
    try:
        from digit_recognition import SpeechTranscriber
    except ImportError as exc:
        raise RuntimeError(
            "Speech transcription dependencies are unavailable. Install requirements.txt with Python 3.11 "
            "or 3.12 before using the app."
        ) from exc
    return SpeechTranscriber(model_size=model_size)


def _transcript_html(result: "TranscriptionResult") -> str:
    transcript = escape(result.text).replace("\n", "<br>")
    return (
        '<div class="transcript-shell">'
        '<div class="transcript-label">Transcript</div>'
        f'<div class="transcript-body">{transcript}</div>'
        "</div>"
    )


def _transcribe(uploaded_file, transcriber: "SpeechTranscriber", language: str | None):
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    try:
        result = transcriber.transcribe_file(temp_path, language=language)
        audio = transcriber.processor.load_audio(temp_path)
        report = transcriber.processor.quality_report(audio)
    finally:
        temp_path.unlink(missing_ok=True)

    return audio, result, report


def _render_about_page() -> None:
    _render_hero(
        kicker="Project Overview",
        title="About",
        copy=(
            "This page explains what the speech-to-text application does, why it was built, "
            "how the solution works end to end, and who built it."
        ),
        pills=[
            ("Project summary", "#about-project"),
            ("System workflow", "#system-workflow"),
            ("About the Author", "#author-profile"),
            ("Future development", "#future-development"),
        ],
    )

    _detail_card(
        "About the project",
        _html_paragraphs(
            [
                (
                    "This project is an interactive speech-to-text application that converts "
                    "spoken audio into written words. A user can either speak directly into the "
                    "browser using a microphone or upload an existing audio file, and the app "
                    "returns a transcript together with a confidence score and supporting diagnostics."
                ),
                (
                    "The goal of the project is to make speech AI practical, understandable, and "
                    "easy to explore. Instead of limiting the user to a narrow classification task, "
                    "the app now handles free-form speech and presents the output in a way that both "
                    "technical and non-technical users can interpret."
                ),
                (
                    "This makes the solution useful for educational demonstrations, AI research, "
                    "speech-interface prototyping, and general experimentation with audio-to-text workflows."
                ),
            ]
        ),
        anchor_id="about-project",
    )

    _detail_card(
        "How the solution works",
        _html_paragraphs(
            [
                (
                    "The solution begins with audio capture. The user either records speech in the "
                    "browser or uploads a supported audio file such as WAV, MP3, M4A, FLAC, or OGG. "
                    "The app loads the audio at a consistent sample rate so the rest of the pipeline "
                    "can work on a stable representation of the signal."
                ),
                (
                    "Once the audio is available, the transcription engine processes the speech and "
                    "decodes it into natural-language text. The underlying model is designed for "
                    "automatic speech recognition, which means it does not try to choose from a small "
                    "set of labels. Instead, it generates actual written language from what it hears."
                ),
                (
                    "After transcription, the app presents the result in a structured way. The main "
                    "transcript shows the recognized text, the confidence score summarizes how reliable "
                    "the transcription appears to be, and language metadata helps the user understand "
                    "what language the system believes it detected."
                ),
                (
                    "The app also exposes segment-level timing, which makes it easier to inspect how "
                    "different parts of the recording were interpreted. In addition, waveform and "
                    "spectrogram views help reveal silence, clipping, or noise, while audio-quality "
                    "checks help explain why a result may be strong, weak, or partially inaccurate."
                ),
                (
                    "Taken together, the solution is not just a transcript generator. It is a clear, "
                    "inspectable speech-processing workflow that helps the user understand both the text "
                    "output and the quality of the underlying audio."
                ),
            ]
        ),
        anchor_id="system-workflow",
    )

    _detail_card(
        "Future development",
        _html_paragraphs(
            [
                (
                    "A dedicated future-development guide is maintained in the repository for anyone "
                    "who wants to extend this work. It outlines realistic enhancement directions, "
                    "implementation considerations, and contributor expectations."
                ),
                (
                    "That document also makes it clear that future work built on this foundation "
                    "must acknowledge the original project originator."
                ),
            ]
        ),
        anchor_id="future-development",
    )
    st.link_button("future development", FUTURE_DEVELOPMENT_URL)

    author_text_col, author_image_col = st.columns([1.45, 1], gap="large")
    with author_text_col:
        _detail_card(
            "About the Author",
            _author_profile_html(),
            anchor_id="author-profile",
        )

    with author_image_col:
        st.markdown("<div style='height: 3.35rem;'></div>", unsafe_allow_html=True)
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
        model_label = st.selectbox(
            "Transcription model",
            ["Fast (tiny)", "Balanced (base)", "Detailed (small)"],
            index=1,
        )
        model_size = {
            "Fast (tiny)": "tiny",
            "Balanced (base)": "base",
            "Detailed (small)": "small",
        }[model_label]
        language_label = st.selectbox(
            "Language hint",
            ["Auto detect", "English"],
            index=0,
        )
        language = None if language_label == "Auto detect" else "en"
        st.markdown("Supported formats: WAV, MP3, M4A, FLAC, OGG")
        st.markdown("---")
        st.caption("The first run may take longer while the speech model downloads.")
        st.caption("Best results come from clear speech and low background noise.")

    _section_intro(
        "Input",
        (
            "Pick how you want to interact with the app. The microphone option works "
            "directly in the browser, while upload mode is handy for pre-recorded test clips."
        ),
        anchor_id="input-section",
    )

    input_method = st.radio(
        "Audio source",
        ["Record with microphone", "Upload audio file"],
        horizontal=True,
    )

    audio_source = None
    if input_method == "Record with microphone":
        st.caption(
            "Click record, allow microphone access in your browser, speak naturally, then stop recording."
        )
        audio_source = st.audio_input(
            "Record speech",
            sample_rate=22050,
            key=f"audio-input-{st.session_state.audio_input_key}",
        )
    else:
        audio_source = st.file_uploader(
            "Upload an audio file containing speech",
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

    try:
        transcriber = _load_transcriber(model_size)
        audio, result, report = _transcribe(audio_source, transcriber, language=language)
    except RuntimeError as exc:
        st.error(str(exc))
        st.caption(
            "If you deploy on Streamlit Community Cloud, choose Python 3.11 or 3.12 in Advanced settings."
        )
        return
    except Exception as exc:
        st.error(f"Transcription failed: {exc}")
        st.caption("If this is the first run, confirm that model downloads are allowed in the deployment environment.")
        return

    st.markdown(
        """
        <div class="results-banner" id="results-section">
            Transcript ready. Review the written output, confidence score, and audio diagnostics below.
        </div>
        """,
        unsafe_allow_html=True,
    )

    transcript_col, metrics_col = st.columns([1.7, 1], gap="large")
    with transcript_col:
        st.markdown(_transcript_html(result), unsafe_allow_html=True)

    with metrics_col:
        st.metric("Confidence", f"{result.confidence:.1%}")
        st.metric("Detected language", result.language.upper())
        st.metric(
            "Language confidence",
            "N/A" if result.language_confidence is None else f"{result.language_confidence:.1%}",
        )
        st.metric("Transcript duration", f"{result.duration_seconds:.1f}s")

    if result.segments:
        _section_intro(
            "Transcript timeline",
            "Each row shows a transcribed segment with timing and confidence.",
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "start_seconds": round(segment.start_seconds, 2),
                        "end_seconds": round(segment.end_seconds, 2),
                        "confidence": round(segment.confidence, 3),
                        "text": segment.text,
                    }
                    for segment in result.segments
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    viz_col, qa_col = st.columns([1.5, 1])
    with viz_col:
        _section_intro(
            "Signal view",
            (
                "The waveform and mel spectrogram help you see whether the recording is clear, "
                "clipped, or dominated by silence."
            ),
        )
        st.pyplot(_plot_audio(audio, transcriber.processor.sample_rate))

    with qa_col:
        _section_intro(
            "Quality checks",
            "These quick diagnostics flag common recording issues that can reduce prediction quality.",
            anchor_id="quality-checks-section",
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
            "transcript_preview": result.text[:80] + ("..." if len(result.text) > 80 else ""),
            "confidence": round(result.confidence, 4),
            "language": result.language,
        }
    )
    _section_intro(
        "Transcription history",
        (
            "Track how the app responds across multiple takes so you can compare speaking styles, "
            "background conditions, and transcript confidence."
        ),
        anchor_id="history-section",
    )
    st.markdown(
        '<div class="history-caption">Latest transcripts from this session appear below.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)


def main() -> None:
    _inject_styles()
    page = _render_sidebar_navigation()

    if page == "About":
        _render_about_page()
    else:
        _render_app_page()

    _render_footer()


if __name__ == "__main__":
    main()
