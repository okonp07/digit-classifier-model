"""Training pipeline for the digit recognition models."""

from __future__ import annotations

import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .audio import AudioProcessor
from .datasets import (
    AudioAugmenter,
    SpeechDigitDataset,
    group_split_records,
    parse_fsdd_records,
    parse_gsc_records,
)
from .model import LightweightDigitCNN
from .predictor import DigitPredictor


FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.9.zip"
SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 32
    step_size: int = 8
    gamma: float = 0.5
    weight_decay: float = 1e-4
    augment_probability: float = 0.5
    max_samples_per_digit: int = 400


def _download_if_missing(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    return destination


def download_fsdd(data_dir: str | Path = "data") -> Path:
    data_root = Path(data_dir)
    archive_path = _download_if_missing(FSDD_URL, data_root / "fsdd.zip")
    recordings_path = data_root / "free-spoken-digit-dataset-1.0.9" / "recordings"
    if not recordings_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_root)
    return recordings_path


def download_speech_commands(data_dir: str | Path = "data") -> Path:
    data_root = Path(data_dir)
    archive_path = _download_if_missing(SPEECH_COMMANDS_URL, data_root / "speech_commands_v0.02.tar.gz")
    extracted_path = data_root / "speech_commands"
    if not extracted_path.exists():
        extracted_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(extracted_path)
    return extracted_path


def prepare_multi_datasets(
    data_dir: str | Path = "data",
    fsdd_path: Optional[str | Path] = None,
    speech_commands_path: Optional[str | Path] = None,
    max_samples_per_digit: int = 400,
    batch_size: int = 32,
    use_augmentation: bool = True,
    augment_probability: float = 0.5,
    download: bool = False,
) -> dict[str, object]:
    data_root = Path(data_dir)
    resolved_fsdd = Path(fsdd_path) if fsdd_path else data_root / "free-spoken-digit-dataset-1.0.9" / "recordings"
    resolved_gsc = Path(speech_commands_path) if speech_commands_path else data_root / "speech_commands"

    if download:
        resolved_fsdd = download_fsdd(data_root)
        resolved_gsc = download_speech_commands(data_root)

    if not resolved_fsdd.exists():
        raise FileNotFoundError(f"FSDD recordings not found at {resolved_fsdd}. Set fsdd_path or use download=True.")
    if not resolved_gsc.exists():
        raise FileNotFoundError(
            f"Speech Commands data not found at {resolved_gsc}. Set speech_commands_path or use download=True."
        )

    processor = AudioProcessor()
    train_records, val_records = group_split_records(
        parse_fsdd_records(resolved_fsdd) + parse_gsc_records(resolved_gsc, max_samples_per_digit=max_samples_per_digit)
    )

    train_dataset = SpeechDigitDataset(
        train_records,
        processor=processor,
        augmenter=AudioAugmenter() if use_augmentation else None,
        augment_probability=augment_probability,
    )
    val_dataset = SpeechDigitDataset(
        val_records,
        processor=processor,
        augmenter=None,
        augment_probability=0.0,
    )

    return {
        "processor": processor,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "train_records": train_records,
        "val_records": val_records,
    }


def _run_epoch(
    model: LightweightDigitCNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[optim.Optimizer] = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(labels.size(0))

    return total_loss / max(total_samples, 1), 100.0 * total_correct / max(total_samples, 1)


def save_checkpoint(
    model: LightweightDigitCNN,
    save_path: str | Path,
    processor: AudioProcessor,
    metrics: dict[str, object],
) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_params": {
                "input_channels": processor.n_mels,
                "num_classes": 10,
            },
            "processor_params": {
                "sample_rate": processor.sample_rate,
                "max_duration": processor.max_duration,
                "n_mels": processor.n_mels,
                "n_fft": processor.n_fft,
                "hop_length": processor.hop_length,
            },
            "training_stats": metrics,
        },
        path,
    )
    return path


def train_enhanced_model(
    datasets: Optional[dict[str, object]] = None,
    epochs: int = 20,
    use_augmentation: bool = True,
    save_path: str | Path = "models/enhanced_digit_model.pth",
    batch_size: int = 32,
    learning_rate: float = 0.001,
    data_dir: str | Path = "data",
    download: bool = False,
) -> tuple[LightweightDigitCNN, dict[str, object]]:
    if datasets is None:
        datasets = prepare_multi_datasets(
            data_dir=data_dir,
            batch_size=batch_size,
            use_augmentation=use_augmentation,
            download=download,
        )

    train_loader = datasets["train_loader"]
    val_loader = datasets["val_loader"]
    processor: AudioProcessor = datasets["processor"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightDigitCNN(input_channels=processor.n_mels, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }
    best_val_accuracy = 0.0

    for _ in range(epochs):
        train_loss, train_accuracy = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

    metrics = {
        "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": history["val_accuracy"][-1],
        "dataset_size": len(datasets["train_dataset"]) + len(datasets["val_dataset"]),
        "epochs": epochs,
        "history": history,
        "config": asdict(
            TrainingConfig(
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                augment_probability=0.5 if use_augmentation else 0.0,
            )
        ),
    }
    save_checkpoint(model, save_path, processor, metrics)
    return model, metrics


def _ensure_predictor(model_or_path: str | Path | DigitPredictor) -> DigitPredictor:
    if isinstance(model_or_path, DigitPredictor):
        return model_or_path
    return DigitPredictor(model_or_path)


def compare_model_performance(
    original_model_path: str | Path = "models/lightweight_digit_model.pth",
    enhanced_model_path: str | Path = "models/enhanced_digit_model.pth",
    test_data_path: str | Path = "real_world_recordings",
) -> dict[str, object]:
    from evaluation import analyze_and_visualize_results, test_real_world_performance

    results = test_real_world_performance(
        recordings_path=test_data_path,
        original_model=_ensure_predictor(original_model_path),
        enhanced_model=_ensure_predictor(enhanced_model_path),
    )
    summary, _ = analyze_and_visualize_results(results, show=False)
    return summary
