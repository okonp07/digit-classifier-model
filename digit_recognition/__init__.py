"""Reusable API for spoken digit recognition."""

from .audio import AudioProcessor, AudioQualityReport

__all__ = [
    "AudioProcessor",
    "AudioQualityReport",
    "DigitPredictor",
    "LightweightDigitCNN",
]


def __getattr__(name: str):
    if name == "DigitPredictor":
        from .predictor import DigitPredictor

        return DigitPredictor
    if name == "LightweightDigitCNN":
        from .model import LightweightDigitCNN

        return LightweightDigitCNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
