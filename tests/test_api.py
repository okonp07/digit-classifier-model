import math
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("torch")

from digit_recognition import DigitPredictor  # noqa: E402
from digit_recognition.audio import AudioProcessor  # noqa: E402
from digit_recognition.model import LightweightDigitCNN  # noqa: E402
from digit_recognition.training import save_checkpoint  # noqa: E402


def _write_wave(path: Path, sample_rate: int = 22_050, duration: float = 1.0) -> None:
    frames = int(sample_rate * duration)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for idx in range(frames):
            value = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * idx / sample_rate))
            wav_file.writeframes(struct.pack("<h", value))


def test_predictor_loads_checkpoint_and_predicts(tmp_path):
    checkpoint_path = tmp_path / "test_model.pth"
    processor = AudioProcessor()
    model = LightweightDigitCNN()
    save_checkpoint(model, checkpoint_path, processor, {"best_val_accuracy": 0.0})

    audio_path = tmp_path / "3.wav"
    _write_wave(audio_path)

    predictor = DigitPredictor(checkpoint_path, device="cpu")
    pred, conf, probs = predictor.predict_from_file(audio_path)

    assert 0 <= pred <= 9
    assert 0.0 <= conf <= 1.0
    assert probs.shape == (10,)
