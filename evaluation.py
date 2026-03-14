"""Evaluation helpers for offline testing and README examples."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from digit_recognition import DigitPredictor


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")


def _ensure_predictor(model_or_path: str | Path | DigitPredictor) -> DigitPredictor:
    if isinstance(model_or_path, DigitPredictor):
        return model_or_path
    return DigitPredictor(model_or_path)


def _extract_true_digit(path: str | Path) -> int | None:
    stem = Path(path).stem.lower()
    for token in re.split(r"[^0-9]+", stem):
        if token in {str(i) for i in range(10)}:
            return int(token)
    return None


def test_real_world_performance(
    recordings_path: str | Path,
    original_model: str | Path | DigitPredictor,
    enhanced_model: str | Path | DigitPredictor,
) -> list[dict[str, object]]:
    base_path = Path(recordings_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Recordings path does not exist: {base_path}")

    original_predictor = _ensure_predictor(original_model)
    enhanced_predictor = _ensure_predictor(enhanced_model)

    audio_files = sorted(
        path for path in base_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in {base_path}")

    results: list[dict[str, object]] = []
    for audio_file in audio_files:
        true_digit = _extract_true_digit(audio_file)
        orig_pred, orig_conf, orig_probs = original_predictor.predict_from_file(audio_file)
        enh_pred, enh_conf, enh_probs = enhanced_predictor.predict_from_file(audio_file)
        results.append(
            {
                "filename": audio_file.name,
                "path": str(audio_file),
                "true_digit": true_digit,
                "orig_pred": orig_pred,
                "orig_conf": orig_conf,
                "orig_probs": orig_probs,
                "orig_correct": None if true_digit is None else orig_pred == true_digit,
                "enh_pred": enh_pred,
                "enh_conf": enh_conf,
                "enh_probs": enh_probs,
                "enh_correct": None if true_digit is None else enh_pred == true_digit,
            }
        )
    return results


def _accuracy(results: Iterable[dict[str, object]], key: str) -> float | None:
    labeled = [row for row in results if row["true_digit"] is not None]
    if not labeled:
        return None
    return 100.0 * sum(1 for row in labeled if row[key]) / len(labeled)


def analyze_and_visualize_results(
    results: list[dict[str, object]],
    show: bool = True,
) -> tuple[dict[str, object], plt.Figure]:
    if not results:
        raise ValueError("No evaluation results supplied.")

    orig_accuracy = _accuracy(results, "orig_correct")
    enh_accuracy = _accuracy(results, "enh_correct")
    agreement_rate = 100.0 * sum(1 for row in results if row["orig_pred"] == row["enh_pred"]) / len(results)
    summary = {
        "num_files": len(results),
        "labeled_files": sum(1 for row in results if row["true_digit"] is not None),
        "original_accuracy": orig_accuracy,
        "enhanced_accuracy": enh_accuracy,
        "agreement_rate": agreement_rate,
        "original_avg_confidence": float(np.mean([row["orig_conf"] for row in results])),
        "enhanced_avg_confidence": float(np.mean([row["enh_conf"] for row in results])),
    }

    figure, axes = plt.subplots(1, 3, figsize=(16, 4))

    conf_data = [
        [row["orig_conf"] for row in results],
        [row["enh_conf"] for row in results],
    ]
    axes[0].boxplot(conf_data, tick_labels=["Original", "Enhanced"])
    axes[0].set_title("Confidence Distribution")
    axes[0].set_ylabel("Confidence")

    if orig_accuracy is not None and enh_accuracy is not None:
        axes[1].bar(["Original", "Enhanced"], [orig_accuracy, enh_accuracy], color=["#3b82f6", "#ef4444"])
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Accuracy on Labeled Audio")
        axes[1].set_ylabel("Accuracy (%)")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No labels in filenames.\nAccuracy unavailable.", ha="center", va="center")

    axes[2].bar(["Agreement"], [agreement_rate], color="#22c55e")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("Model Agreement")
    axes[2].set_ylabel("Rate (%)")

    figure.tight_layout()
    if show:
        plt.show()
    return summary, figure
