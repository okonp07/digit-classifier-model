"""Audio preprocessing utilities shared by training, evaluation, and the app."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Optional, Sequence

import numpy as np


DEFAULT_SAMPLE_RATE = 22_050
DEFAULT_MAX_DURATION = 1.0
DEFAULT_N_MFCC = 13
DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 256
EXPECTED_TIME_FRAMES = 87


def _require_librosa():
    cache_dir = Path(tempfile.gettempdir()) / "digit_classifier_numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised in real runtime only
        raise ImportError(
            "librosa is required for audio processing. Install project requirements first."
        ) from exc
    return librosa


@dataclass(frozen=True)
class AudioQualityReport:
    duration_seconds: float
    peak_amplitude: float
    rms_amplitude: float
    issues: tuple[str, ...]


class AudioProcessor:
    """Load audio, normalize duration, and extract MFCC features."""

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_duration: float = DEFAULT_MAX_DURATION,
        n_mels: int = DEFAULT_N_MFCC,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
    ) -> None:
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_length = int(max_duration * sample_rate)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, path: str | Path) -> np.ndarray:
        librosa = _require_librosa()
        audio, _ = librosa.load(Path(path), sr=self.sample_rate, mono=True)
        return audio.astype(np.float32, copy=False)

    def pad_or_trim(self, audio: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=-1)
        if len(array) > self.max_length:
            start = (len(array) - self.max_length) // 2
            array = array[start : start + self.max_length]
        elif len(array) < self.max_length:
            padding = self.max_length - len(array)
            array = np.pad(array, (0, padding), mode="constant")
        return array.astype(np.float32, copy=False)

    def extract_mfcc(self, audio: Sequence[float] | np.ndarray) -> np.ndarray:
        librosa = _require_librosa()
        processed = self.pad_or_trim(audio)
        mfcc = librosa.feature.mfcc(
            y=processed,
            sr=self.sample_rate,
            n_mfcc=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        if mfcc.shape[1] != EXPECTED_TIME_FRAMES:
            mfcc = librosa.util.fix_length(mfcc, size=EXPECTED_TIME_FRAMES, axis=1)
        return mfcc.astype(np.float32, copy=False)

    def load_and_preprocess(self, path: str | Path) -> np.ndarray:
        return self.extract_mfcc(self.load_audio(path))

    def quality_report(
        self,
        audio: Sequence[float] | np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> AudioQualityReport:
        array = np.asarray(audio, dtype=np.float32)
        sr = sample_rate or self.sample_rate
        duration = float(len(array) / sr) if sr else 0.0
        peak = float(np.max(np.abs(array))) if len(array) else 0.0
        rms = float(np.sqrt(np.mean(np.square(array)))) if len(array) else 0.0

        issues: list[str] = []
        if duration < 0.5:
            issues.append("Audio is shorter than 0.5 seconds.")
        if peak < 0.01:
            issues.append("Audio appears very quiet.")
        if peak > 0.99:
            issues.append("Audio may be clipping.")

        return AudioQualityReport(
            duration_seconds=duration,
            peak_amplitude=peak,
            rms_amplitude=rms,
            issues=tuple(issues),
        )
