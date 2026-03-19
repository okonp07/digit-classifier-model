"""Reusable API for spoken digit recognition."""

from .audio import AudioProcessor, AudioQualityReport

__all__ = [
    "AudioProcessor",
    "AudioQualityReport",
    "DigitPredictor",
    "LightweightDigitCNN",
    "SpeechTranscriber",
]


def __getattr__(name: str):
    if name == "DigitPredictor":
        from .predictor import DigitPredictor

        return DigitPredictor
    if name == "LightweightDigitCNN":
        from .model import LightweightDigitCNN

        return LightweightDigitCNN
    if name == "SpeechTranscriber":
        from .transcriber import SpeechTranscriber

        return SpeechTranscriber
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
