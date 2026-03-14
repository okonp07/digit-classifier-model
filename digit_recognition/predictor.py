"""Inference helpers and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from .audio import AudioProcessor
from .model import LightweightDigitCNN


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models"


class DigitPredictor:
    """Load a trained model and provide convenience prediction methods."""

    def __init__(
        self,
        model_path: str | Path = "enhanced_digit_model.pth",
        device: Optional[str] = None,
    ) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.device = self._resolve_device(device)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        self.model = LightweightDigitCNN(**checkpoint["model_params"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.processor = AudioProcessor(**checkpoint["processor_params"])
        self.training_stats = checkpoint.get("training_stats", {})

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _candidate_paths(model_path: str | Path) -> Iterable[Path]:
        path = Path(model_path)
        yield path
        yield DEFAULT_MODEL_DIR / path.name
        yield REPO_ROOT / path.name

    @classmethod
    def _resolve_model_path(cls, model_path: str | Path) -> Path:
        for candidate in cls._candidate_paths(model_path):
            if candidate.exists():
                return candidate.resolve()
        searched = ", ".join(str(path) for path in cls._candidate_paths(model_path))
        raise FileNotFoundError(f"Could not find model checkpoint. Looked in: {searched}")

    def _predict_tensor(self, mfcc: np.ndarray) -> tuple[int, float, np.ndarray]:
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(mfcc_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])
        return predicted_digit, confidence, probabilities

    def _predict_from_audio(self, audio_array: np.ndarray) -> tuple[int, float, np.ndarray]:
        clips = self.processor.inference_clips(audio_array)
        mfcc_batch = np.stack([self.processor.extract_mfcc(clip) for clip in clips], axis=0)
        mfcc_tensor = torch.tensor(mfcc_batch, dtype=torch.float32).to(self.device)

        with torch.inference_mode():
            logits = self.model(mfcc_tensor)
            mean_logits = logits.mean(dim=0, keepdim=True)
            probabilities = torch.softmax(mean_logits, dim=1).squeeze(0).cpu().numpy()

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])
        return predicted_digit, confidence, probabilities

    def predict_from_file(self, audio_path: str | Path) -> tuple[int, float, np.ndarray]:
        audio = self.processor.load_audio(audio_path)
        return self._predict_from_audio(audio)

    def predict_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: Optional[int] = None,
    ) -> tuple[int, float, np.ndarray]:
        if sample_rate and sample_rate != self.processor.sample_rate:
            import librosa

            audio_array = librosa.resample(
                np.asarray(audio_array, dtype=np.float32),
                orig_sr=sample_rate,
                target_sr=self.processor.sample_rate,
            )
        return self._predict_from_audio(np.asarray(audio_array, dtype=np.float32))

    def metadata(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "training_stats": self.training_stats,
        }
