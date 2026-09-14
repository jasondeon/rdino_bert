from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_metrics import intraclass_correlation_2_1


ARCHITECTURES = ("mean", "attention")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train recording-level MIL heads on cached WavLM-Large windows"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("outputs/wavlm-large-layer-audit")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/wavlm-large-mil")
    )
    parser.add_argument("--layers", type=int, nargs="+", default=(16, 17, 18, 19, 20, 21))
    parser.add_argument("--madrs-threshold", type=float, default=20.0)
    parser.add_argument("--include-temporal-std", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--attention-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-windows", type=int, default=32)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=(40, 41, 42))
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {name}")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(
    input_dir: Path,
    split: str,
    layers: list[int],
    include_temporal_std: bool,
) -> tuple[np.ndarray, pd.DataFrame, list[np.ndarray]]:
    means = np.load(input_dir / f"{split}_layer_means.npy", mmap_mode="r")
    stds = np.load(input_dir / f"{split}_layer_stds.npy", mmap_mode="r")
    windows = pd.read_csv(
        input_dir / f"{split}_windows.csv", dtype={"subject_id": str}
    )
    if means.ndim != 3 or stds.shape != means.shape or len(windows) != len(means):
        raise ValueError(f"Invalid or misaligned {split} WavLM cache")
    if min(layers) < 0 or max(layers) >= means.shape[1]:
        raise ValueError(f"Requested layers are outside the {means.shape[1]} cached states")
    if not np.array_equal(windows["embedding_index"], np.arange(len(windows))):
        raise ValueError(f"Nonsequential {split} embedding indices")

    # Adjacent WavLM layers share a residual coordinate system. A fixed average
    # avoids choosing a validation-favored layer or fitting an underidentified
    # all-layer mixture on this small cohort.
    mean_features = np.zeros((len(means), means.shape[2]), dtype=np.float32)
    for layer in layers:
        mean_features += np.asarray(means[:, layer, :], dtype=np.float32)
    mean_features /= len(layers)
    if include_temporal_std:
        std_features = np.zeros_like(mean_features)
        for layer in layers:
            std_features += np.asarray(stds[:, layer, :], dtype=np.float32)
        std_features /= len(layers)
        features = np.concatenate((mean_features, std_features), axis=1)
    else:
        features = mean_features

    rows = []
    indices_by_recording = []
    for recording_index, group in windows.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(int)
        scores = group["regression_label"].to_numpy(float)
        if not np.allclose(scores, scores[0]):
            raise ValueError(f"Labels differ within recording {recording_index}")
        first = group.iloc[0]
        rows.append(
            {
                "recording_index": int(recording_index),
                "audio_path": str(first["audio_path"]),
                "subject_id": str(first["subject_id"]),
                "regression_truth": float(scores[0]),
                "window_count": len(indices),
            }
        )
        indices_by_recording.append(indices)
    return features, pd.DataFrame(rows), indices_by_recording


class RecordingMIL(nn.Module):
    def __init__(
        self,
        input_size: int,
        projection_size: int,
        attention_size: int,
        dropout: float,
        architecture: str,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.architecture = architecture
        self.input_normalization = nn.LayerNorm(input_size, elementwise_affine=False)
        self.projection = nn.Sequential(
            nn.Linear(input_size, projection_size),
            nn.LayerNorm(projection_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if architecture == "attention":
            self.attention_tanh = nn.Linear(projection_size, attention_size)
            self.attention_gate = nn.Linear(projection_size, attention_size)
            self.attention_score = nn.Linear(attention_size, 1)
        self.output_dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(projection_size, 1)
        self.classification_head = nn.Linear(projection_size, 1)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.projection(self.input_normalization(features))
        if self.architecture == "attention":
            scores = self.attention_score(
                torch.tanh(self.attention_tanh(hidden))
                * torch.sigmoid(self.attention_gate(hidden))
            ).squeeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=1)
        else:
            weights = mask.to(hidden.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = torch.einsum("bw,bwd->bd", weights, hidden)
        pooled = self.output_dropout(pooled)
        regression = self.regression_head(pooled).squeeze(-1)
        classification = self.classification_head(pooled).squeeze(-1)
        return regression, classification, weights


def make_batch(
    features: np.ndarray,
    indices_by_recording: list[np.ndarray],
    recording_indices: np.ndarray,
    *,
    device: torch.device,
    max_windows: int | None,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
    selected = []
    for recording_index in recording_indices:
        indices = indices_by_recording[int(recording_index)]
        if max_windows is not None and len(indices) > max_windows:
            if generator is None:
                raise ValueError("Training window sampling requires a generator")
            positions = torch.randperm(len(indices), generator=generator)[:max_windows].numpy()
            indices = indices[positions]
        selected.append(indices)
    longest = max(len(indices) for indices in selected)
    batch = np.zeros((len(selected), longest, features.shape[1]), dtype=np.float32)
    mask = np.zeros((len(selected), longest), dtype=bool)
    for row, indices in enumerate(selected):
        batch[row, : len(indices)] = features[indices]
        mask[row, : len(indices)] = True
    return (
        torch.from_numpy(batch).to(device, non_blocking=True),
        torch.from_numpy(mask).to(device, non_blocking=True),
        selected,
    )


def loss_components(
    regression: torch.Tensor,
    classification: torch.Tensor,
    regression_truth: torch.Tensor,
    classification_truth: torch.Tensor,
    positive_weight: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    regression_loss = nn.functional.mse_loss(regression, regression_truth)
    classification_loss = nn.functional.binary_cross_entropy_with_logits(
        classification, classification_truth, pos_weight=positive_weight
    )
    total = (
        args.regression_weight * regression_loss
        + args.classification_weight * classification_loss
    )
    return total, regression_loss, classification_loss


def train_epoch(
    model: RecordingMIL,
    optimizer: torch.optim.Optimizer,
    features: np.ndarray,
    indices_by_recording: list[np.ndarray],
    recording_indices: np.ndarray,
    standardized_targets: torch.Tensor,
    binary_targets: torch.Tensor,
    positive_weight: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    model.train()
    order = recording_indices[
        torch.randperm(len(recording_indices), generator=generator).numpy()
    ]
    totals = np.zeros(3, dtype=float)
    count = 0
    for start in range(0, len(order), args.batch_size):
        batch_indices = order[start : start + args.batch_size]
        batch, mask, _ = make_batch(
            features,
            indices_by_recording,
            batch_indices,
            device=device,
            max_windows=args.max_train_windows,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        regression, classification, _ = model(batch, mask)
        losses = loss_components(
            regression,
            classification,
            standardized_targets[batch_indices],
            binary_targets[batch_indices],
            positive_weight,
            args,
        )
        losses[0].backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        totals += np.asarray([float(value.detach()) for value in losses]) * len(batch_indices)
        count += len(batch_indices)
    return dict(zip(("loss", "regression_loss", "classification_loss"), totals / count))


def predict(
    model: RecordingMIL,
    features: np.ndarray,
    indices_by_recording: list[np.ndarray],
    recording_indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    model.eval()
    regression_rows = []
    probability_rows = []
    attentions: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(recording_indices), batch_size):
            batch_indices = recording_indices[start : start + batch_size]
            batch, mask, selected = make_batch(
                features,
                indices_by_recording,
                batch_indices,
                device=device,
                max_windows=None,
                generator=None,
            )
            regression, classification, weights = model(batch, mask)
            regression_rows.append(regression.cpu().numpy())
            probability_rows.append(classification.sigmoid().cpu().numpy())
            weight_array = weights.cpu().numpy()
            for row, (recording_index, window_indices) in enumerate(
                zip(batch_indices, selected)
            ):
                attentions[int(recording_index)] = weight_array[row, : len(window_indices)]
    return (
        np.concatenate(regression_rows),
        np.concatenate(probability_rows),
        attentions,
    )


def make_targets(
    regression_truth: np.ndarray,
    binary_truth: np.ndarray,
    fit_indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor]:
    target_mean = float(regression_truth[fit_indices].mean())
    target_std = float(regression_truth[fit_indices].std(ddof=0))
    if target_std <= 0:
        raise ValueError("Regression target has zero variance")
    standardized = torch.as_tensor(
        (regression_truth - target_mean) / target_std,
        dtype=torch.float32,
        device=device,
    )
    binary = torch.as_tensor(binary_truth, dtype=torch.float32, device=device)
    fit_binary = binary_truth[fit_indices]
    counts = np.bincount(fit_binary, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Training fold contains only one binary class")
    positive_weight = torch.tensor(
        counts[0] / counts[1], dtype=torch.float32, device=device
    )
    return standardized, binary, target_mean, target_std, positive_weight


def evaluate_loss(
    model: RecordingMIL,
    features: np.ndarray,
    indices_by_recording: list[np.ndarray],
    indices: np.ndarray,
    standardized_targets: torch.Tensor,
    binary_targets: torch.Tensor,
    positive_weight: torch.Tensor,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    regression, probability, _ = predict(
        model,
        features,
        indices_by_recording,
        indices,
        batch_size=args.batch_size,
        device=device,
    )
    regression_tensor = torch.as_tensor(regression, device=device)
    probability_tensor = torch.as_tensor(probability, device=device).clamp(1e-6, 1 - 1e-6)
    classification_logits = torch.logit(probability_tensor)
    losses = loss_components(
        regression_tensor,
        classification_logits,
        standardized_targets[indices],
        binary_targets[indices],
        positive_weight,
        args,
    )
    return dict(
        zip(("loss", "regression_loss", "classification_loss"), [float(x) for x in losses])
    )


def fit_with_early_stopping(
    architecture: str,
    features: np.ndarray,
    indices_by_recording: list[np.ndarray],
    regression_truth: np.ndarray,
    binary_truth: np.ndarray,
    fit_indices: np.ndarray,
    held_out_indices: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[RecordingMIL, int, np.ndarray, np.ndarray, list[dict[str, float]]]:
    set_seed(seed)
    model = RecordingMIL(
        features.shape[1],
        args.projection_size,
        args.attention_size,
        args.dropout,
        architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    standardized, binary, target_mean, target_std, positive_weight = make_targets(
        regression_truth, binary_truth, fit_indices, device
    )
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    without_improvement = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        train_values = train_epoch(
            model,
            optimizer,
            features,
            indices_by_recording,
            fit_indices,
            standardized,
            binary,
            positive_weight,
            args=args,
            device=device,
            generator=generator,
        )
        validation_values = evaluate_loss(
            model,
            features,
            indices_by_recording,
            held_out_indices,
            standardized,
            binary,
            positive_weight,
            args=args,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_values.items()},
                **{f"validation_{key}": value for key, value in validation_values.items()},
            }
        )
        if validation_values["loss"] < best_loss - args.minimum_delta:
            best_loss = validation_values["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Early stopping never saved a model")
    model.load_state_dict(best_state)
    regression, probability, _ = predict(
        model,
        features,
        indices_by_recording,
        held_out_indices,
        batch_size=args.batch_size,
        device=device,
    )
    return model, best_epoch, regression * target_std + target_mean, probability, history


def fit_full(
    architecture: str,
    features: np.ndarray,
    indices_by_recording: list[np.ndarray],
    regression_truth: np.ndarray,
    binary_truth: np.ndarray,
    epochs: int,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[RecordingMIL, float, float, list[dict[str, float]]]:
    set_seed(seed)
    model = RecordingMIL(
        features.shape[1],
        args.projection_size,
        args.attention_size,
        args.dropout,
        architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    indices = np.arange(len(regression_truth))
    standardized, binary, target_mean, target_std, positive_weight = make_targets(
        regression_truth, binary_truth, indices, device
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        values = train_epoch(
            model,
            optimizer,
            features,
            indices_by_recording,
            indices,
            standardized,
            binary,
            positive_weight,
            args=args,
            device=device,
            generator=generator,
        )
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in values.items()}})
    return model, target_mean, target_std, history


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "icc_2_1": intraclass_correlation_2_1(truth, prediction),
        "pearson": float(np.corrcoef(truth, prediction)[0, 1]),
        "mean_error": float(np.mean(prediction - truth)),
        "prediction_target_sd_ratio": float(np.std(prediction, ddof=1) / np.std(truth, ddof=1)),
    }


def choose_cutoff(truth: np.ndarray, probability: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(truth, probability)
    finite = np.isfinite(thresholds)
    scores = true_positive[finite] - false_positive[finite]
    candidates = thresholds[finite][np.isclose(scores, scores.max())]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def classification_metrics(
    truth: np.ndarray, probability: np.ndarray, cutoff: float
) -> dict[str, float]:
    prediction = (probability >= cutoff).astype(int)
    return {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "average_precision": float(average_precision_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "sensitivity": float(recall_score(truth, prediction)),
        "specificity": float(np.mean(prediction[truth == 0] == 0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "cutoff": cutoff,
    }


def clustered_bootstrap(
    frame: pd.DataFrame,
    prediction: str,
    metric: str,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    groups = {
        subject: np.asarray(indices, dtype=int)
        for subject, indices in frame.groupby("subject_id").groups.items()
    }
    subjects = np.asarray(list(groups), dtype=object)
    truth_column = "regression_truth" if metric == "r2" else "binary_truth"
    truth = frame[truth_column].to_numpy()
    values = frame[prediction].to_numpy(float)
    rng = np.random.default_rng(seed)
    results = []
    function = r2_score if metric == "r2" else roc_auc_score
    for _ in range(samples):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        indices = np.concatenate([groups[subject] for subject in sampled])
        if len(np.unique(truth[indices])) > 1:
            results.append(function(truth[indices], values[indices]))
    return [float(value) for value in np.percentile(results, [2.5, 97.5])]


def paired_bootstrap(
    frame: pd.DataFrame,
    first: str,
    second: str,
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    groups = {
        subject: np.asarray(indices, dtype=int)
        for subject, indices in frame.groupby("subject_id").groups.items()
    }
    subjects = np.asarray(list(groups), dtype=object)
    truth_column = "regression_truth" if metric == "r2" else "binary_truth"
    truth = frame[truth_column].to_numpy()
    first_values = frame[first].to_numpy(float)
    second_values = frame[second].to_numpy(float)
    function = r2_score if metric == "r2" else roc_auc_score
    observed = float(function(truth, first_values) - function(truth, second_values))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        indices = np.concatenate([groups[subject] for subject in sampled])
        if len(np.unique(truth[indices])) > 1:
            differences.append(
                function(truth[indices], first_values[indices])
                - function(truth[indices], second_values[indices])
            )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": observed, "ci_low": float(low), "ci_high": float(high)}


def save_attention(
    output_path: Path,
    windows: pd.DataFrame,
    indices_by_recording: list[np.ndarray],
    attention_by_seed: list[dict[int, np.ndarray]],
) -> None:
    rows = []
    for recording_position, indices in enumerate(indices_by_recording):
        weights = np.stack(
            [attention[recording_position] for attention in attention_by_seed]
        ).mean(axis=0)
        subset = windows.iloc[indices]
        for (_, window), weight in zip(subset.iterrows(), weights):
            rows.append(
                {
                    "embedding_index": int(window["embedding_index"]),
                    "recording_index": int(window["recording_index"]),
                    "audio_path": str(window["audio_path"]),
                    "subject_id": str(window["subject_id"]),
                    "window_start": float(window["window_start"]),
                    "window_end": float(window["window_end"]),
                    "attention_weight": float(weight),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_summary(metrics: dict[str, Any], output_path: Path) -> None:
    architectures = list(ARCHITECTURES)
    oof_r2 = [metrics[name]["oof"]["regression"]["r2"] for name in architectures]
    val_r2 = [metrics[name]["validation"]["regression"]["r2"] for name in architectures]
    oof_auc = [metrics[name]["oof"]["classification"]["roc_auc"] for name in architectures]
    val_auc = [metrics[name]["validation"]["classification"]["roc_auc"] for name in architectures]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    x = np.arange(len(architectures))
    width = 0.35
    axes[0].bar(x - width / 2, oof_r2, width, label="Training OOF")
    axes[0].bar(x + width / 2, val_r2, width, label="External validation")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Regression", ylabel="R²", xticks=x, xticklabels=architectures)
    axes[1].bar(x - width / 2, oof_auc, width, label="Training OOF")
    axes[1].bar(x + width / 2, val_auc, width, label="External validation")
    axes[1].axhline(0.5, color="black", linewidth=1)
    axes[1].set(title="MADRS binary classification", ylabel="AUROC", xticks=x, xticklabels=architectures)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2 or not args.seeds or args.bootstrap_samples < 100:
        raise ValueError("Use at least 2 folds, one seed, and 100 bootstrap samples")
    if args.batch_size < 1 or args.max_train_windows < 1:
        raise ValueError("Batch size and maximum training windows must be positive")
    if args.regression_weight < 0 or args.classification_weight < 0:
        raise ValueError("Loss weights cannot be negative")
    if args.regression_weight + args.classification_weight <= 0:
        raise ValueError("At least one loss must be enabled")
    device = resolve_device(args.device)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extraction = json.loads((input_dir / "extraction_config.json").read_text())
    layer_names = list(extraction["layer_names"])
    if extraction["model_name"] != "microsoft/wavlm-large":
        raise ValueError(f"Expected microsoft/wavlm-large, found {extraction['model_name']}")
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": extraction["model_name"],
        "layers": args.layers,
        "layer_names": [layer_names[index] for index in args.layers],
        "madrs_threshold": args.madrs_threshold,
        "include_temporal_std": args.include_temporal_std,
        "projection_size": args.projection_size,
        "attention_size": args.attention_size,
        "dropout": args.dropout,
        "regression_weight": args.regression_weight,
        "classification_weight": args.classification_weight,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size_recordings": args.batch_size,
        "max_train_windows_per_recording": args.max_train_windows,
        "validation_windows": "all",
        "cv_folds": args.cv_folds,
        "seeds": args.seeds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "minimum_delta": args.minimum_delta,
        "gradient_clip": args.gradient_clip,
        "bootstrap_samples": args.bootstrap_samples,
        "device": str(device),
        "external_validation_used_for_selection": False,
        "epoch_selection": "median grouped-CV best optimizer updates per seed",
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; cached features were not loaded and training did not run.")
        return

    train_features, train_recordings, train_indices = load_features(
        input_dir, "train", list(args.layers), args.include_temporal_std
    )
    validation_features, validation_recordings, validation_indices = load_features(
        input_dir, "validation", list(args.layers), args.include_temporal_std
    )
    train_windows = pd.read_csv(input_dir / "train_windows.csv", dtype={"subject_id": str})
    validation_windows = pd.read_csv(
        input_dir / "validation_windows.csv", dtype={"subject_id": str}
    )
    if set(train_recordings["subject_id"]) & set(validation_recordings["subject_id"]):
        raise ValueError("Training and external validation share subjects")
    regression_truth = train_recordings["regression_truth"].to_numpy(float)
    validation_regression_truth = validation_recordings["regression_truth"].to_numpy(float)
    binary_truth = (regression_truth >= args.madrs_threshold).astype(int)
    validation_binary_truth = (
        validation_regression_truth >= args.madrs_threshold
    ).astype(int)
    groups = train_recordings["subject_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seeds[0]
    )
    folds = list(splitter.split(regression_truth, binary_truth, groups))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    all_metrics: dict[str, Any] = {}
    cv_rows = []
    history_rows = []
    validation_frame = validation_recordings.copy()
    validation_frame["binary_truth"] = validation_binary_truth
    validation_attention: list[dict[int, np.ndarray]] = []
    for architecture in ARCHITECTURES:
        oof_regression_by_seed = []
        oof_probability_by_seed = []
        validation_regression_by_seed = []
        validation_probability_by_seed = []
        architecture_attention = []
        for seed in args.seeds:
            oof_regression = np.full(len(train_recordings), np.nan)
            oof_probability = np.full(len(train_recordings), np.nan)
            best_epochs = []
            fold_fit_sizes = []
            for fold, (fit_indices, held_out_indices) in enumerate(folds, start=1):
                model, best_epoch, fold_regression, fold_probability, history = (
                    fit_with_early_stopping(
                        architecture,
                        train_features,
                        train_indices,
                        regression_truth,
                        binary_truth,
                        fit_indices,
                        held_out_indices,
                        args=args,
                        device=device,
                        seed=seed * 100 + fold,
                    )
                )
                oof_regression[held_out_indices] = fold_regression
                oof_probability[held_out_indices] = fold_probability
                best_epochs.append(best_epoch)
                fold_fit_sizes.append(len(fit_indices))
                fold_values = {
                    **{f"regression_{key}": value for key, value in regression_metrics(regression_truth[held_out_indices], fold_regression).items()},
                    "classification_roc_auc": float(roc_auc_score(binary_truth[held_out_indices], fold_probability)),
                }
                cv_rows.append(
                    {"architecture": architecture, "seed": seed, "fold": fold, "best_epoch": best_epoch, **fold_values}
                )
                history_rows.extend(
                    {"architecture": architecture, "seed": seed, "fold": fold, **row}
                    for row in history
                )
                print(
                    f"architecture={architecture} seed={seed} fold={fold} "
                    f"best_epoch={best_epoch} r2={fold_values['regression_r2']:.4f} "
                    f"auc={fold_values['classification_roc_auc']:.4f}",
                    flush=True,
                )
            oof_regression_by_seed.append(oof_regression)
            oof_probability_by_seed.append(oof_probability)
            median_epoch = float(np.median(best_epochs))
            fold_steps = float(
                np.median([math.ceil(size / args.batch_size) for size in fold_fit_sizes])
            )
            full_steps = math.ceil(len(train_recordings) / args.batch_size)
            final_epochs = max(1, int(round(median_epoch * fold_steps / full_steps)))
            model, target_mean, target_std, history = fit_full(
                architecture,
                train_features,
                train_indices,
                regression_truth,
                binary_truth,
                final_epochs,
                args=args,
                device=device,
                seed=seed,
            )
            validation_standardized, validation_probability, attention = predict(
                model,
                validation_features,
                validation_indices,
                np.arange(len(validation_recordings)),
                batch_size=args.batch_size,
                device=device,
            )
            validation_regression_by_seed.append(
                validation_standardized * target_std + target_mean
            )
            validation_probability_by_seed.append(validation_probability)
            architecture_attention.append(attention)
            history_rows.extend(
                {"architecture": architecture, "seed": seed, "fold": 0, **row}
                for row in history
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": architecture,
                    "input_size": train_features.shape[1],
                    "projection_size": args.projection_size,
                    "attention_size": args.attention_size,
                    "dropout": args.dropout,
                    "layers": args.layers,
                    "include_temporal_std": args.include_temporal_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "madrs_threshold": args.madrs_threshold,
                    "epochs": final_epochs,
                    "seed": seed,
                },
                output_dir / f"model_{architecture}_seed_{seed}.pt",
            )
            print(
                f"architecture={architecture} seed={seed} full_epochs={final_epochs}",
                flush=True,
            )

        oof_regression = np.stack(oof_regression_by_seed, axis=1).mean(axis=1)
        oof_probability = np.stack(oof_probability_by_seed, axis=1).mean(axis=1)
        validation_regression = np.stack(validation_regression_by_seed, axis=1).mean(axis=1)
        validation_probability = np.stack(validation_probability_by_seed, axis=1).mean(axis=1)
        cutoff = choose_cutoff(binary_truth, oof_probability)
        oof_frame = train_recordings.copy()
        oof_frame["binary_truth"] = binary_truth
        oof_frame["regression_prediction"] = oof_regression
        oof_frame["classification_probability"] = oof_probability
        oof_frame["classification_prediction"] = (oof_probability >= cutoff).astype(int)
        oof_frame.to_csv(output_dir / f"oof_predictions_{architecture}.csv", index=False)
        validation_frame[f"regression_{architecture}"] = validation_regression
        validation_frame[f"probability_{architecture}"] = validation_probability
        validation_frame[f"prediction_{architecture}"] = (
            validation_probability >= cutoff
        ).astype(int)
        all_metrics[architecture] = {
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "oof": {
                "regression": regression_metrics(regression_truth, oof_regression),
                "classification": classification_metrics(binary_truth, oof_probability, cutoff),
            },
            "validation": {
                "regression": regression_metrics(validation_regression_truth, validation_regression),
                "classification": classification_metrics(validation_binary_truth, validation_probability, cutoff),
            },
        }
        all_metrics[architecture]["validation"]["regression"]["r2_ci"] = clustered_bootstrap(
            validation_frame, f"regression_{architecture}", "r2", samples=args.bootstrap_samples, seed=args.seeds[0] + len(all_metrics)
        )
        all_metrics[architecture]["validation"]["classification"]["roc_auc_ci"] = clustered_bootstrap(
            validation_frame, f"probability_{architecture}", "auc", samples=args.bootstrap_samples, seed=args.seeds[0] + 100 + len(all_metrics)
        )
        if architecture == "attention":
            validation_attention = architecture_attention

    validation_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / "cv_folds.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
    save_attention(
        output_dir / "validation_attention_weights.csv",
        validation_windows,
        validation_indices,
        validation_attention,
    )
    all_metrics["paired_attention_minus_mean"] = {
        "regression_r2": paired_bootstrap(
            validation_frame, "regression_attention", "regression_mean", "r2", samples=args.bootstrap_samples, seed=args.seeds[0] + 200
        ),
        "classification_roc_auc": paired_bootstrap(
            validation_frame, "probability_attention", "probability_mean", "auc", samples=args.bootstrap_samples, seed=args.seeds[0] + 201
        ),
    }
    all_metrics["configuration"] = config
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
    plot_summary(all_metrics, output_dir / "model_comparison.png")

    lines = [
        "# Frozen WavLM-Large recording MIL experiment",
        "",
        "The fixed mean-pooling and gated-attention models use identical fixed averages",
        f"of cached layers {args.layers}, joint standardized MADRS regression and MADRS",
        f"≥{args.madrs_threshold:g} classification, and subject-grouped training CV.",
        "External validation did not control epochs, cutoffs, or any model choice.",
        "",
        "| Pooling | OOF R² | Validation R² [95% CI] | OOF AUROC | Validation AUROC [95% CI] | Validation macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for architecture in ARCHITECTURES:
        values = all_metrics[architecture]
        regression_ci = values["validation"]["regression"]["r2_ci"]
        auc_ci = values["validation"]["classification"]["roc_auc_ci"]
        lines.append(
            f"| {architecture} | {values['oof']['regression']['r2']:.3f} | "
            f"{values['validation']['regression']['r2']:.3f} [{regression_ci[0]:.3f}, {regression_ci[1]:.3f}] | "
            f"{values['oof']['classification']['roc_auc']:.3f} | "
            f"{values['validation']['classification']['roc_auc']:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] | "
            f"{values['validation']['classification']['macro_f1']:.3f} |"
        )
    regression_delta = all_metrics["paired_attention_minus_mean"]["regression_r2"]
    auc_delta = all_metrics["paired_attention_minus_mean"]["classification_roc_auc"]
    lines.extend(
        [
            "",
            "## Paired external-validation comparison",
            "",
            f"Attention minus mean pooling: ΔR² {regression_delta['difference']:.3f} "
            f"[{regression_delta['ci_low']:.3f}, {regression_delta['ci_high']:.3f}]; "
            f"ΔAUROC {auc_delta['difference']:.3f} "
            f"[{auc_delta['ci_low']:.3f}, {auc_delta['ci_high']:.3f}].",
            "",
            "Confidence intervals resample validation subjects as clusters. Training uses",
            f"at most {args.max_train_windows} random windows per recording per epoch;",
            "validation and inference use every available window.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
