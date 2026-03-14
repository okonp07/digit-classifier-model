"""Dataset and augmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from .audio import AudioProcessor


@dataclass(frozen=True)
class AudioRecord:
    path: Path
    label: int
    source: str
    group: str


class AudioAugmenter:
    """Applies light augmentation for better robustness."""

    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        noise = np.random.normal(0, noise_factor, audio.shape).astype(np.float32)
        return audio + noise

    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        import librosa

        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio: np.ndarray, sr: int, n_steps: int = 0) -> np.ndarray:
        import librosa

        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    def augment_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        aug_type = np.random.choice(
            ["noise", "stretch", "pitch", "none"],
            p=[0.3, 0.2, 0.2, 0.3],
        )
        if aug_type == "noise":
            return self.add_noise(audio, noise_factor=float(np.random.uniform(0.001, 0.01)))
        if aug_type == "stretch":
            return self.time_stretch(audio, rate=float(np.random.uniform(0.9, 1.1)))
        if aug_type == "pitch":
            return self.pitch_shift(audio, sr, n_steps=int(np.random.randint(-2, 3)))
        return audio


class SpeechDigitDataset(Dataset):
    def __init__(
        self,
        records: Iterable[AudioRecord],
        processor: AudioProcessor,
        augmenter: Optional[AudioAugmenter] = None,
        augment_probability: float = 0.5,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.augmenter = augmenter
        self.augment_probability = augment_probability

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        audio = self.processor.load_audio(record.path)
        if self.augmenter and np.random.random() < self.augment_probability:
            audio = self.augmenter.augment_audio(audio, self.processor.sample_rate)
        mfcc = self.processor.extract_mfcc(audio)
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(record.label, dtype=torch.long)


def parse_fsdd_records(recordings_path: str | Path) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    for path in sorted(Path(recordings_path).glob("*.wav")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        digit, speaker = int(parts[0]), parts[1]
        records.append(AudioRecord(path=path, label=digit, source="fsdd", group=f"fsdd:{speaker}"))
    return records


def parse_gsc_records(
    speech_commands_path: str | Path,
    digit_mapping: Optional[dict[str, int]] = None,
    max_samples_per_digit: Optional[int] = None,
) -> list[AudioRecord]:
    mapping = digit_mapping or {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
    }
    records: list[AudioRecord] = []
    root = Path(speech_commands_path)
    for word, digit in mapping.items():
        files = sorted((root / word).glob("*.wav"))
        if max_samples_per_digit is not None:
            files = files[:max_samples_per_digit]
        for path in files:
            speaker = path.stem.split("_nohash_")[0]
            records.append(AudioRecord(path=path, label=digit, source="gsc", group=f"gsc:{speaker}"))
    return records


def group_split_records(
    records: list[AudioRecord],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    indices = np.arange(len(records))
    groups = np.array([record.group for record in records])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(indices, groups=groups))
    return [records[i] for i in train_idx], [records[i] for i in val_idx]
