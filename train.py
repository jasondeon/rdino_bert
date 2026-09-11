from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer

from augmentation import WaveformAugmenter
from dataset import MultimodalCollator, MultimodalDataset
from evaluation_metrics import intraclass_correlation_2_1
from model import BertRdinoModel


class RecordingBalancedSampler(Sampler[int]):
    """Select grouped random windows per recording, then shuffle recordings."""

    def __init__(
        self, dataset: MultimodalDataset, seed: int, windows_per_recording: int
    ) -> None:
        if windows_per_recording < 1:
            raise ValueError("windows_per_recording must be at least 1")
        indices_by_recording: dict[int, list[int]] = defaultdict(list)
        for window_index, window in enumerate(dataset.windows):
            indices_by_recording[window.recording_index].append(window_index)
        self.indices_by_recording = list(indices_by_recording.values())
        self.windows_per_recording = windows_per_recording
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        order = torch.randperm(
            len(self.indices_by_recording), generator=self.generator
        ).tolist()
        selected: list[int] = []
        for recording_index in order:
            indices = self.indices_by_recording[recording_index]
            if len(indices) >= self.windows_per_recording:
                choices = torch.randperm(len(indices), generator=self.generator)[
                    : self.windows_per_recording
                ].tolist()
            else:
                choices = torch.randint(
                    len(indices),
                    (self.windows_per_recording,),
                    generator=self.generator,
                ).tolist()
            selected.extend(indices[index] for index in choices)
        return iter(selected)

    def __len__(self) -> int:
        return len(self.indices_by_recording) * self.windows_per_recording


class RecordingBalancedWindowSampler(Sampler[int]):
    """Give each recording equal sampling weight without changing the loss unit."""

    def __init__(self, dataset: MultimodalDataset, seed: int) -> None:
        indices_by_recording: dict[int, list[int]] = defaultdict(list)
        for window_index, window in enumerate(dataset.windows):
            indices_by_recording[window.recording_index].append(window_index)
        if not indices_by_recording:
            raise ValueError("Recording-balanced sampling requires eligible recordings")
        self.indices_by_recording = list(indices_by_recording.values())
        self.samples_per_epoch = len(dataset)
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        recording_count = len(self.indices_by_recording)
        base_count, remainder = divmod(self.samples_per_epoch, recording_count)
        extra_recordings = set(
            torch.randperm(recording_count, generator=self.generator)[:remainder].tolist()
        )
        selected: list[int] = []
        for recording_index, indices in enumerate(self.indices_by_recording):
            sample_count = base_count + int(recording_index in extra_recordings)
            if sample_count <= len(indices):
                choices = torch.randperm(len(indices), generator=self.generator)[
                    :sample_count
                ].tolist()
            else:
                choices = torch.randperm(
                    len(indices), generator=self.generator
                ).tolist()
                choices.extend(
                    torch.randint(
                        len(indices),
                        (sample_count - len(indices),),
                        generator=self.generator,
                    ).tolist()
                )
            selected.extend(indices[index] for index in choices)
        order = torch.randperm(len(selected), generator=self.generator).tolist()
        return iter(selected[index] for index in order)

    def __len__(self) -> int:
        return self.samples_per_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT + RDINO on a CSV manifest")
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--rdino-yaml", type=Path, default=Path("assets/rdino.yaml"))
    parser.add_argument("--rdino-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--text-model", default="mental/mental-bert-base-uncased")
    parser.add_argument(
        "--modality",
        choices=("both", "text", "audio"),
        default="both",
        help="Use both modalities, text only, or audio only (default: both)",
    )
    parser.add_argument(
        "--embedding-normalization",
        choices=("batchnorm", "layernorm", "l2", "none"),
        default="batchnorm",
        help="Normalization applied to each modality embedding",
    )
    parser.add_argument(
        "--fusion-architecture",
        choices=("two_layer", "single_layer"),
        default="two_layer",
        help=(
            "Two-layer fusion MLP or the colleague-style single Linear+SiLU "
            "head (default: two_layer)"
        ),
    )
    parser.add_argument(
        "--disable-text-lora",
        action="store_true",
        help="Keep the text backbone fully frozen without text LoRA adapters",
    )
    parser.add_argument(
        "--disable-rdino-lora",
        action="store_true",
        help="Keep the RDINO backbone fully frozen without RDINO LoRA adapters",
    )
    parser.add_argument(
        "--rdino-lora-scope",
        choices=("terminal", "mfa_pooling", "all_pointwise"),
        default="terminal",
        help=(
            "Adapt terminal pooling/output convolutions, include MFA, or adapt "
            "all shape-safe 1x1 RDINO Conv1d layers (default: terminal)"
        ),
    )
    parser.add_argument(
        "--update-rdino-batchnorm-stats",
        action="store_true",
        help=(
            "Allow frozen RDINO BatchNorm running statistics to update; by default "
            "they remain fixed at their pretrained values"
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate this many batches before each optimizer update (default: 1)",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--train-sampling",
        choices=("recording", "balanced_window", "window"),
        default="recording",
        help=(
            "Use grouped recording-level loss (recording), equal recording sampling "
            "with window-level loss (balanced_window), or every window (window)"
        ),
    )
    parser.add_argument(
        "--train-windows-per-recording",
        type=int,
        default=4,
        help="Windows averaged for each recording-level training loss (default: 4)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--text-learning-rate",
        type=float,
        help="Optional text-adapter LR; defaults to --learning-rate",
    )
    parser.add_argument(
        "--audio-learning-rate",
        type=float,
        help="Optional RDINO-adapter LR; defaults to --learning-rate",
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        help="Optional normalization/fusion/head LR; defaults to --learning-rate",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--lr-scheduler",
        choices=("plateau", "none"),
        default="plateau",
        help="Learning-rate schedule (default: plateau on validation R-squared)",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help=(
            "Stop after this many validation events without recording-level "
            "R-squared improvement"
        ),
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help=(
            "Run full validation every N epochs; scheduler and early-stopping "
            "patience count validation events (default: 1)"
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=55.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument(
        "--eligibility-window-seconds",
        type=float,
        help=(
            "Require recordings to support this window length while allowing "
            "--window-seconds to vary; useful for comparable HPO cohorts"
        ),
    )
    parser.add_argument(
        "--speaker-gap-policy",
        choices=("preserve", "concatenate"),
        default="preserve",
        help=(
            "Preserve unlabeled time between consecutive primary-speaker segments "
            "or remove it by concatenation (default: preserve)"
        ),
    )
    parser.add_argument(
        "--stride-jitter-seconds",
        type=float,
        default=0.0,
        help="Randomly shift each training window start by up to this many seconds",
    )
    parser.add_argument("--include-empty-text", action="store_true")
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument(
        "--augment-audio",
        action="store_true",
        help="Apply training-only AWGN and synthetic room reverb",
    )
    parser.add_argument("--awgn-probability", type=float, default=0.5)
    parser.add_argument("--awgn-snr-min-db", type=float, default=10.0)
    parser.add_argument("--awgn-snr-max-db", type=float, default=30.0)
    parser.add_argument("--reverb-probability", type=float, default=0.3)
    parser.add_argument("--reverb-rt60-min-seconds", type=float, default=0.2)
    parser.add_argument("--reverb-rt60-max-seconds", type=float, default=0.8)
    parser.add_argument("--reverb-wet-min", type=float, default=0.1)
    parser.add_argument("--reverb-wet-max", type=float, default=0.4)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument(
        "--class-weighting",
        choices=("sqrt_inverse_frequency", "none"),
        default="sqrt_inverse_frequency",
        help="Per-class weighting for classification losses (default: sqrt inverse frequency)",
    )
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument(
        "--log-task-gradients",
        action="store_true",
        help="Log classification/regression gradient norms and cosine once per epoch",
    )
    parser.add_argument(
        "--task-gradient-batches-per-epoch",
        type=int,
        default=1,
        help="Number of leading training batches to diagnose per epoch (default: 1)",
    )
    parser.add_argument(
        "--standardize-regression-labels",
        action="store_true",
        help="Fit mean/std on training-recording labels and train in standardized units",
    )
    parser.add_argument("--seed", type=int, default=40)
    return parser.parse_args()


def move_batch(
    batch: dict, device: torch.device
) -> tuple[dict, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    tokens = {name: value.to(device, non_blocking=True) for name, value in batch["tokens"].items()}
    waveforms = batch["waveforms"]
    return (
        tokens,
        waveforms.to(device, non_blocking=True) if waveforms is not None else None,
        batch["class_labels"].to(device, non_blocking=True),
        batch["regression_labels"].to(device, non_blocking=True),
    )


def _regression_standardization(train_data, enabled: bool) -> dict[str, float | bool]:
    if not enabled:
        return {"enabled": False, "mean": 0.0, "std": 1.0}
    labels = np.asarray(
        [recording.regression_label for recording in train_data.recordings], dtype=np.float64
    )
    mean = float(labels.mean())
    std = float(labels.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Cannot standardize regression labels because training std is zero")
    return {"enabled": True, "mean": mean, "std": std}


def _classification_weighting(
    train_data: MultimodalDataset, num_classes: int, sampling: str, method: str
) -> dict[str, str | list[int] | list[float]]:
    if sampling in {"recording", "balanced_window"}:
        eligible_recordings = sorted(
            {window.recording_index for window in train_data.windows}
        )
        labels = [
            train_data.recordings[index].class_label for index in eligible_recordings
        ]
        sampling_unit = "eligible_recordings"
    else:
        labels = [
            train_data.recordings[window.recording_index].class_label
            for window in train_data.windows
        ]
        sampling_unit = "windows"

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if len(counts) != num_classes or np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Cannot weight classification because classes are absent: {missing}")
    if method == "sqrt_inverse_frequency":
        weights = 1.0 / np.sqrt(counts)
        weights /= np.average(weights, weights=counts)
    elif method == "none":
        weights = np.ones(num_classes, dtype=np.float64)
    else:
        raise ValueError(f"Unknown class-weighting method: {method}")
    return {
        "method": method,
        "sampling_unit": sampling_unit,
        "loss_normalization": "mean_of_weighted_sample_losses",
        "counts": counts.astype(int).tolist(),
        "weights": weights.tolist(),
    }


def _standardize_regression(
    labels: torch.Tensor, standardization: dict[str, float | bool]
) -> torch.Tensor:
    return (labels - float(standardization["mean"])) / float(standardization["std"])


def _restore_regression_scale(
    predictions: torch.Tensor, standardization: dict[str, float | bool]
) -> torch.Tensor:
    return (
        predictions * float(standardization["std"])
        + float(standardization["mean"])
    )


def _optimizer_parameter_groups(
    model: BertRdinoModel, args: argparse.Namespace
) -> list[dict]:
    """Separate backbone adapters from task heads for discriminative LRs."""
    learning_rates = {
        "text": (
            args.text_learning_rate
            if args.text_learning_rate is not None
            else args.learning_rate
        ),
        "audio": (
            args.audio_learning_rate
            if args.audio_learning_rate is not None
            else args.learning_rate
        ),
        "head": (
            args.head_learning_rate
            if args.head_learning_rate is not None
            else args.learning_rate
        ),
    }
    parameters: dict[str, list[torch.nn.Parameter]] = {
        "text": [],
        "audio": [],
        "head": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("text_model."):
            parameters["text"].append(parameter)
        elif name.startswith("rdino_backbone."):
            parameters["audio"].append(parameter)
        else:
            parameters["head"].append(parameter)

    groups = []
    for group_name in ("text", "audio", "head"):
        if parameters[group_name]:
            groups.append(
                {
                    "params": parameters[group_name],
                    "lr": learning_rates[group_name],
                    "group_name": group_name,
                }
            )
    return groups


def _optimizer_learning_rates(optimizer: AdamW) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _primary_learning_rate(learning_rates: dict[str, float]) -> float:
    """Keep the historic scalar column useful for plots and old summaries."""
    if "head" in learning_rates:
        return learning_rates["head"]
    return next(iter(learning_rates.values()))


def task_gradient_diagnostics(
    model: BertRdinoModel,
    classification_loss: torch.Tensor,
    regression_loss: torch.Tensor,
    classification_weight: float,
    regression_weight: float,
    epoch: int,
    batch_number: int,
) -> list[dict[str, float | int | str | bool]]:
    """Measure task gradient magnitude and alignment on shared trainable parameters."""
    groups = model.shared_trainable_parameter_groups()
    parameters: list[torch.nn.Parameter] = []
    group_indices: dict[str, list[int]] = {}
    for group_name, group_parameters in groups.items():
        indices: list[int] = []
        for parameter in group_parameters:
            indices.append(len(parameters))
            parameters.append(parameter)
        group_indices[group_name] = indices
    if not parameters:
        return []

    classification_gradients = torch.autograd.grad(
        classification_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    regression_gradients = torch.autograd.grad(
        regression_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def summarize(group_name: str, indices: list[int]):
        device = parameters[0].device
        class_squared = torch.zeros((), device=device)
        regression_squared = torch.zeros((), device=device)
        dot_product = torch.zeros((), device=device)
        for index in indices:
            class_gradient = classification_gradients[index]
            regression_gradient = regression_gradients[index]
            if class_gradient is not None:
                class_squared += class_gradient.detach().float().square().sum()
            if regression_gradient is not None:
                regression_squared += regression_gradient.detach().float().square().sum()
            if class_gradient is not None and regression_gradient is not None:
                dot_product += (
                    class_gradient.detach().float()
                    * regression_gradient.detach().float()
                ).sum()
        class_norm = math.sqrt(class_squared.item())
        regression_norm = math.sqrt(regression_squared.item())
        denominator = class_norm * regression_norm
        cosine = dot_product.item() / denominator if denominator > 0 else float("nan")
        weighted_class_norm = classification_weight * class_norm
        weighted_regression_norm = regression_weight * regression_norm
        weighted_ratio = (
            weighted_class_norm / weighted_regression_norm
            if weighted_regression_norm > 0
            else float("nan")
        )
        return {
            "epoch": epoch,
            "batch": batch_number,
            "parameter_group": group_name,
            "parameter_count": sum(parameters[index].numel() for index in indices),
            "classification_gradient_norm": class_norm,
            "regression_gradient_norm": regression_norm,
            "weighted_classification_gradient_norm": weighted_class_norm,
            "weighted_regression_gradient_norm": weighted_regression_norm,
            "weighted_class_to_regression_norm_ratio": weighted_ratio,
            "cosine_similarity": cosine,
            "gradient_conflict": bool(cosine < 0),
        }

    rows = [summarize("all_shared", list(range(len(parameters))))]
    rows.extend(
        summarize(group_name, indices)
        for group_name, indices in group_indices.items()
    )
    return rows


def aggregate_recording_batch(
    class_logits: torch.Tensor,
    regression_outputs: torch.Tensor,
    class_labels: torch.Tensor,
    regression_labels: torch.Tensor,
    recording_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average window predictions and return one prediction/target per recording."""
    if len(recording_indices) != class_logits.shape[0]:
        raise ValueError("recording_indices and model outputs have different lengths")
    recording_ids = torch.as_tensor(recording_indices, device=class_logits.device)
    unique_ids, inverse = torch.unique(recording_ids, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(unique_ids)).to(class_logits.dtype)

    probabilities = class_logits.softmax(dim=1)
    probability_sums = torch.zeros(
        len(unique_ids), probabilities.shape[1],
        device=probabilities.device, dtype=probabilities.dtype,
    ).index_add(0, inverse, probabilities)
    mean_probabilities = probability_sums / counts.unsqueeze(1)
    regression_sums = torch.zeros(
        len(unique_ids), device=regression_outputs.device, dtype=regression_outputs.dtype
    ).index_add(0, inverse, regression_outputs)
    mean_regression_outputs = regression_sums / counts

    recording_class_labels: list[torch.Tensor] = []
    recording_regression_labels: list[torch.Tensor] = []
    for group_index in range(len(unique_ids)):
        members = inverse == group_index
        group_class_labels = class_labels[members]
        group_regression_labels = regression_labels[members]
        if not torch.all(group_class_labels == group_class_labels[0]):
            raise ValueError("Class labels differ within a recording")
        if not torch.all(group_regression_labels == group_regression_labels[0]):
            raise ValueError("Regression labels differ within a recording")
        recording_class_labels.append(group_class_labels[0])
        recording_regression_labels.append(group_regression_labels[0])

    return (
        mean_probabilities,
        mean_regression_outputs,
        torch.stack(recording_class_labels),
        torch.stack(recording_regression_labels),
    )


def evaluate(
    model,
    loader,
    device,
    output_path: Path,
    standardization: dict[str, float | bool],
    class_weights: list[float],
    classification_weight: float = 1.0,
    regression_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    rows: list[dict] = []
    probability_columns: list[str] = []
    with torch.no_grad():
        for batch in loader:
            tokens, waveforms, class_labels, regression_labels = move_batch(batch, device)
            class_logits, standardized_regression_output = model(tokens, waveforms)
            class_probabilities = class_logits.softmax(dim=1).cpu().numpy()
            if not probability_columns:
                probability_columns = [
                    f"class_probability_{index}"
                    for index in range(class_probabilities.shape[1])
                ]
            standardized_regression_labels = _standardize_regression(
                regression_labels, standardization
            )
            regression_output = _restore_regression_scale(
                standardized_regression_output, standardization
            )
            for index in range(class_labels.shape[0]):
                row = {
                    "recording_index": batch["recording_indices"][index],
                    "audio_path": batch["audio_paths"][index],
                    "subject_id": batch["subject_ids"][index],
                    "speaker": batch["speakers"][index],
                    "window_start": batch["window_starts"][index],
                    "window_end": batch["window_ends"][index],
                    "class_truth": class_labels[index].item(),
                    "class_prediction": class_probabilities[index].argmax().item(),
                    "regression_truth": regression_labels[index].item(),
                    "regression_prediction": regression_output[index].item(),
                    "regression_truth_standardized": (
                        standardized_regression_labels[index].item()
                    ),
                    "regression_prediction_standardized": (
                        standardized_regression_output[index].item()
                    ),
                }
                row.update(dict(zip(probability_columns, class_probabilities[index])))
                rows.append(row)

    window_frame = pd.DataFrame(rows)
    window_frame.to_csv(output_path, index=False)
    grouped = window_frame.groupby("recording_index", sort=False)
    recording_frame = grouped.agg(
        audio_path=("audio_path", "first"),
        subject_id=("subject_id", "first"),
        speaker=("speaker", "first"),
        class_truth=("class_truth", "first"),
        regression_truth=("regression_truth", "first"),
        regression_prediction=("regression_prediction", "mean"),
        regression_truth_standardized=("regression_truth_standardized", "first"),
        regression_prediction_standardized=(
            "regression_prediction_standardized", "mean"
        ),
        window_count=("audio_path", "size"),
    )
    recording_frame = recording_frame.join(grouped[probability_columns].mean())
    mean_probabilities = recording_frame[probability_columns].to_numpy()
    recording_frame["class_prediction"] = mean_probabilities.argmax(axis=1)
    recording_frame = recording_frame.reset_index()
    recording_output_path = output_path.with_name(f"recording_{output_path.name}")
    recording_frame.to_csv(recording_output_path, index=False)

    class_truth = recording_frame["class_truth"].to_numpy(dtype=int)
    class_predictions = recording_frame["class_prediction"].to_numpy(dtype=int)
    standardized_truth = recording_frame["regression_truth_standardized"].to_numpy()
    standardized_predictions = recording_frame[
        "regression_prediction_standardized"
    ].to_numpy()
    regression_truth = recording_frame["regression_truth"].to_numpy()
    regression_predictions = recording_frame["regression_prediction"].to_numpy()
    negative_log_likelihood = -np.log(
        mean_probabilities[np.arange(len(class_truth)), class_truth].clip(1e-12)
    )
    validation_class_weights = np.asarray(class_weights, dtype=np.float64)[class_truth]
    classification_loss = float(
        np.mean(negative_log_likelihood * validation_class_weights)
    )
    regression_loss = float(mean_squared_error(standardized_truth, standardized_predictions))
    regression_rmse = float(np.sqrt(regression_loss))
    regression_rmse_original_scale = float(
        np.sqrt(mean_squared_error(regression_truth, regression_predictions))
    )
    regression_icc_2_1 = intraclass_correlation_2_1(
        regression_truth, regression_predictions
    )
    return {
        "classification_loss": classification_loss,
        "regression_loss": regression_loss,
        "total_loss": (
            classification_weight * classification_loss
            + regression_weight * regression_loss
        ),
        "accuracy": float(accuracy_score(class_truth, class_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(class_truth, class_predictions)),
        "macro_f1": float(f1_score(class_truth, class_predictions, average="macro")),
        "regression_rmse": regression_rmse,
        "regression_rmse_original_scale": regression_rmse_original_scale,
        "regression_r2": float(r2_score(standardized_truth, standardized_predictions)),
        "regression_icc_2_1": regression_icc_2_1,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.lora_rank < 1:
        raise ValueError("--lora-rank must be at least 1")
    if args.lora_alpha < 1:
        raise ValueError("--lora-alpha must be at least 1")
    learning_rate_arguments = {
        "--learning-rate": args.learning_rate,
        "--text-learning-rate": args.text_learning_rate,
        "--audio-learning-rate": args.audio_learning_rate,
        "--head-learning-rate": args.head_learning_rate,
    }
    for option, value in learning_rate_arguments.items():
        if value is not None and value <= 0:
            raise ValueError(f"{option} must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.validation_interval < 1:
        raise ValueError("--validation-interval must be at least 1")
    if (
        args.eligibility_window_seconds is not None
        and args.eligibility_window_seconds <= 0
    ):
        raise ValueError("--eligibility-window-seconds must be positive")
    if args.task_gradient_batches_per_epoch < 1:
        raise ValueError("--task-gradient-batches-per-epoch must be at least 1")
    if args.modality == "text" and args.augment_audio:
        raise ValueError("--augment-audio has no effect with --modality text")
    serialized_arguments = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    (args.output_dir / "training_arguments.json").write_text(
        json.dumps(serialized_arguments, indent=2) + "\n", encoding="utf-8"
    )
    augmentation_config = {
        "enabled": args.augment_audio,
        "awgn_probability": args.awgn_probability,
        "awgn_snr_min_db": args.awgn_snr_min_db,
        "awgn_snr_max_db": args.awgn_snr_max_db,
        "reverb_probability": args.reverb_probability,
        "reverb_rt60_min_seconds": args.reverb_rt60_min_seconds,
        "reverb_rt60_max_seconds": args.reverb_rt60_max_seconds,
        "reverb_wet_min": args.reverb_wet_min,
        "reverb_wet_max": args.reverb_wet_max,
    }
    (args.output_dir / "audio_augmentation.json").write_text(
        json.dumps(augmentation_config, indent=2) + "\n", encoding="utf-8"
    )
    audio_augmenter = (
        WaveformAugmenter(
            sample_rate=16_000,
            awgn_probability=args.awgn_probability,
            awgn_snr_min_db=args.awgn_snr_min_db,
            awgn_snr_max_db=args.awgn_snr_max_db,
            reverb_probability=args.reverb_probability,
            reverb_rt60_min_seconds=args.reverb_rt60_min_seconds,
            reverb_rt60_max_seconds=args.reverb_rt60_max_seconds,
            reverb_wet_min=args.reverb_wet_min,
            reverb_wet_max=args.reverb_wet_max,
        )
        if args.augment_audio
        else None
    )
    if audio_augmenter is not None:
        print(f"Training audio augmentation: {json.dumps(augmentation_config)}")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    collator = MultimodalCollator(tokenizer, args.max_text_tokens)
    common = dict(
        sample_rate=16_000,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        num_classes=args.num_classes,
        include_empty_text=args.include_empty_text,
        load_audio=args.modality != "text",
        eligibility_window_seconds=args.eligibility_window_seconds,
        speaker_gap_policy=args.speaker_gap_policy,
    )
    train_data = MultimodalDataset(
        args.train_manifest,
        window_jitter_seconds=args.stride_jitter_seconds,
        **common,
    )
    validation_data = MultimodalDataset(args.validation_manifest, **common)
    if args.stride_jitter_seconds > 0:
        print(
            "Training window-start jitter: "
            f"+/-{args.stride_jitter_seconds:g} seconds"
        )
    classification_weighting = _classification_weighting(
        train_data, args.num_classes, args.train_sampling, args.class_weighting
    )
    (args.output_dir / "classification_weighting.json").write_text(
        json.dumps(classification_weighting, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Classification weighting ({classification_weighting['sampling_unit']}): "
        f"counts={classification_weighting['counts']}, "
        f"weights={[round(value, 6) for value in classification_weighting['weights']]}"
    )
    regression_standardization = _regression_standardization(
        train_data, args.standardize_regression_labels
    )
    (args.output_dir / "regression_standardization.json").write_text(
        json.dumps(regression_standardization, indent=2) + "\n", encoding="utf-8"
    )
    if regression_standardization["enabled"]:
        print(
            "Regression-label standardization fitted on training recordings: "
            f"mean={regression_standardization['mean']:.8g}, "
            f"std={regression_standardization['std']:.8g}"
        )
    if len(train_data) < 2:
        raise ValueError("Training requires at least two examples because the fusion uses BatchNorm")
    print(
        f"Expanded {len(train_data.recordings)} training recordings into "
        f"{len(train_data)} windows (skipped {train_data.skipped_short_recordings} short); "
        f"{len(validation_data.recordings)} validation recordings into "
        f"{len(validation_data)} windows (skipped "
        f"{validation_data.skipped_short_recordings} short)"
    )
    recording_level_training = args.train_sampling == "recording"
    balanced_window_training = args.train_sampling == "balanced_window"
    if recording_level_training:
        if args.train_windows_per_recording < 1:
            raise ValueError("--train-windows-per-recording must be at least 1")
        if args.batch_size % args.train_windows_per_recording != 0:
            raise ValueError(
                "--batch-size must be divisible by --train-windows-per-recording "
                "so a recording is not split across optimizer batches"
            )
        train_sampler = RecordingBalancedSampler(
            train_data, args.seed, args.train_windows_per_recording
        )
        recordings_per_batch = args.batch_size // args.train_windows_per_recording
    elif balanced_window_training:
        train_sampler = RecordingBalancedWindowSampler(train_data, args.seed)
        recordings_per_batch = None
    else:
        train_sampler = None
        recordings_per_batch = None
    samples_per_epoch = len(train_sampler) if train_sampler is not None else len(train_data)
    batches_per_epoch = math.ceil(samples_per_epoch / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    print(
        f"Training sampling: {args.train_sampling}; "
        f"{samples_per_epoch} windows per epoch"
    )
    print(
        f"Gradient accumulation: {args.gradient_accumulation_steps}; "
        f"approximately {optimizer_steps_per_epoch} optimizer steps per epoch"
    )
    if args.modality == "text":
        print("Text-only data path: waveform loading and GPU transfer disabled")
    if recording_level_training:
        print(
            f"Recording-level loss: {args.train_windows_per_recording} windows per "
            f"recording; {recordings_per_batch} recordings per batch; "
            f"{recordings_per_batch * args.gradient_accumulation_steps} "
            "recordings per optimizer update"
        )
    elif balanced_window_training:
        eligible_recordings = len(train_sampler.indices_by_recording)
        mean_windows = len(train_sampler) / eligible_recordings
        print(
            "Balanced window-level loss: approximately "
            f"{mean_windows:.2f} sampled windows per eligible recording"
        )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=samples_per_epoch % args.batch_size == 1,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    model_config = {
        "text_model_name": args.text_model,
        "rdino_yaml": str(args.rdino_yaml.resolve()),
        "rdino_checkpoint": str(args.rdino_checkpoint.resolve()),
        "audio_embedding_dim": 512,
        "fusion_hidden_dim": 200,
        "fusion_dim": 50,
        "num_classes": args.num_classes,
        "dropout": 0.1,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "sample_rate": 16_000,
        "modality": args.modality,
        "use_text_lora": not args.disable_text_lora,
        "use_rdino_lora": not args.disable_rdino_lora,
        "rdino_lora_scope": args.rdino_lora_scope,
        "embedding_normalization": args.embedding_normalization,
        "fusion_architecture": args.fusion_architecture,
        "freeze_rdino_batchnorm_stats": not args.update_rdino_batchnorm_stats,
    }
    model = BertRdinoModel(**model_config).to(device)
    trainable, total = model.trainable_parameter_counts()
    print(f"Device: {device}; trainable parameters: {trainable:,}/{total:,}")
    print(f"Model configuration: {json.dumps(model_config)}")

    class_weight_tensor = torch.tensor(
        classification_weighting["weights"], dtype=torch.float32, device=device
    )
    classification_loss = torch.nn.CrossEntropyLoss(
        weight=class_weight_tensor, reduction="none"
    )
    recording_classification_loss = torch.nn.NLLLoss(
        weight=class_weight_tensor, reduction="none"
    )
    regression_loss = torch.nn.MSELoss()
    optimizer_groups = _optimizer_parameter_groups(model, args)
    optimizer = AdamW(optimizer_groups, weight_decay=args.weight_decay)
    initial_learning_rates = _optimizer_learning_rates(optimizer)
    for group in optimizer.param_groups:
        parameter_count = sum(parameter.numel() for parameter in group["params"])
        print(
            f"Optimizer group {group['group_name']}: "
            f"lr={group['lr']:.8g}; parameters={parameter_count:,}"
        )
    if not 0.0 < args.lr_scheduler_factor < 1.0:
        raise ValueError("--lr-scheduler-factor must be between 0 and 1")
    if args.lr_scheduler_patience < 0:
        raise ValueError("--lr-scheduler-patience cannot be negative")
    if not 0.0 <= args.min_learning_rate <= min(initial_learning_rates.values()):
        raise ValueError(
            "--min-learning-rate must be between 0 and the smallest active "
            "optimizer-group learning rate"
        )
    scheduler_config = {
        "name": args.lr_scheduler,
        "monitor": "validation_regression_r2",
        "mode": "max",
        "factor": args.lr_scheduler_factor,
        "patience": args.lr_scheduler_patience,
        "threshold": args.early_stopping_min_delta,
        "min_learning_rate": args.min_learning_rate,
        "initial_group_learning_rates": initial_learning_rates,
    }
    (args.output_dir / "lr_scheduler.json").write_text(
        json.dumps(scheduler_config, indent=2) + "\n", encoding="utf-8"
    )
    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            threshold=args.early_stopping_min_delta,
            threshold_mode="abs",
            min_lr=args.min_learning_rate,
        )
        if args.lr_scheduler == "plateau"
        else None
    )
    print(f"Learning-rate scheduler: {json.dumps(scheduler_config)}")

    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1")
    history: list[dict[str, float | int]] = []
    gradient_rows: list[dict[str, float | int | str | bool]] = []
    best_validation_r2 = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        epoch_learning_rates = _optimizer_learning_rates(optimizer)
        epoch_learning_rate = _primary_learning_rate(epoch_learning_rates)
        model.train()
        total_losses, classification_losses, regression_losses = [], [], []
        optimizer.zero_grad(set_to_none=True)
        for batch_number, batch in enumerate(train_loader, start=1):
            tokens, waveforms, class_labels, regression_labels = move_batch(batch, device)
            if audio_augmenter is not None and waveforms is not None:
                waveforms = audio_augmenter(waveforms)
            class_logits, regression_output = model(tokens, waveforms)
            if recording_level_training:
                (
                    class_probabilities,
                    regression_output,
                    class_labels,
                    regression_labels,
                ) = aggregate_recording_batch(
                    class_logits,
                    regression_output,
                    class_labels,
                    regression_labels,
                    batch["recording_indices"],
                )
                batch_classification_loss = recording_classification_loss(
                    class_probabilities.clamp_min(1e-12).log(), class_labels
                ).mean()
            else:
                batch_classification_loss = classification_loss(
                    class_logits, class_labels
                ).mean()
            regression_targets = _standardize_regression(
                regression_labels, regression_standardization
            )
            batch_regression_loss = regression_loss(regression_output, regression_targets)
            loss = (
                args.classification_weight * batch_classification_loss
                + args.regression_weight * batch_regression_loss
            )
            if (
                args.log_task_gradients
                and batch_number <= args.task_gradient_batches_per_epoch
            ):
                epoch_gradient_rows = task_gradient_diagnostics(
                    model,
                    batch_classification_loss,
                    batch_regression_loss,
                    args.classification_weight,
                    args.regression_weight,
                    epoch + 1,
                    batch_number,
                )
                gradient_rows.extend(epoch_gradient_rows)
                all_shared = next(
                    (
                        row
                        for row in epoch_gradient_rows
                        if row["parameter_group"] == "all_shared"
                    ),
                    None,
                )
                if all_shared is not None:
                    print(
                        "task_gradients "
                        f"cosine={all_shared['cosine_similarity']:.6f} "
                        "weighted_class_to_regression_ratio="
                        f"{all_shared['weighted_class_to_regression_norm_ratio']:.6f}"
                    )
            accumulation_start = (
                (batch_number - 1) // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            accumulation_group_size = min(
                args.gradient_accumulation_steps,
                len(train_loader) - accumulation_start,
            )
            (loss / accumulation_group_size).backward()
            update_parameters = (
                batch_number % args.gradient_accumulation_steps == 0
                or batch_number == len(train_loader)
            )
            if update_parameters:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_losses.append(loss.item())
            classification_losses.append(batch_classification_loss.item())
            regression_losses.append(batch_regression_loss.item())
            if batch_number == 1 or batch_number % 10 == 0:
                print(f"epoch={epoch + 1} batch={batch_number} loss={loss.item():.6f}")

        print(f"epoch={epoch + 1} mean_train_loss={np.mean(total_losses):.6f}")
        if gradient_rows:
            pd.DataFrame(gradient_rows).to_csv(
                args.output_dir / "gradient_diagnostics.csv", index=False
            )
        should_validate = (
            (epoch + 1) % args.validation_interval == 0
            or epoch + 1 == args.epochs
        )
        if not should_validate:
            continue

        prediction_path = args.output_dir / f"predictions_epoch_{epoch + 1}.csv"
        metrics = evaluate(
            model,
            validation_loader,
            device,
            prediction_path,
            regression_standardization,
            class_weights=classification_weighting["weights"],
            classification_weight=args.classification_weight,
            regression_weight=args.regression_weight,
        )
        if scheduler is not None:
            scheduler.step(metrics["regression_r2"])
        next_learning_rates = _optimizer_learning_rates(optimizer)
        next_learning_rate = _primary_learning_rate(next_learning_rates)
        reduced_groups = [
            group_name
            for group_name, learning_rate in epoch_learning_rates.items()
            if next_learning_rates[group_name] < learning_rate
        ]
        if reduced_groups:
            print(
                "Reduced learning rates: "
                + ", ".join(
                    f"{group_name} {epoch_learning_rates[group_name]:.8g} -> "
                    f"{next_learning_rates[group_name]:.8g}"
                    for group_name in reduced_groups
                )
            )
        history.append({
            "epoch": epoch + 1,
            "learning_rate": epoch_learning_rate,
            "next_learning_rate": next_learning_rate,
            "text_learning_rate": epoch_learning_rates.get("text", float("nan")),
            "audio_learning_rate": epoch_learning_rates.get("audio", float("nan")),
            "head_learning_rate": epoch_learning_rates.get("head", float("nan")),
            "next_text_learning_rate": next_learning_rates.get(
                "text", float("nan")
            ),
            "next_audio_learning_rate": next_learning_rates.get(
                "audio", float("nan")
            ),
            "next_head_learning_rate": next_learning_rates.get(
                "head", float("nan")
            ),
            "train_total_loss": float(np.mean(total_losses)),
            "train_classification_loss": float(np.mean(classification_losses)),
            "train_regression_loss": float(np.mean(regression_losses)),
            "validation_total_loss": metrics["total_loss"],
            "validation_classification_loss": metrics["classification_loss"],
            "validation_regression_loss": metrics["regression_loss"],
            "validation_accuracy": metrics["accuracy"],
            "validation_balanced_accuracy": metrics["balanced_accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "validation_regression_rmse": metrics["regression_rmse"],
            "validation_regression_rmse_original_scale": metrics[
                "regression_rmse_original_scale"
            ],
            "validation_regression_r2": metrics["regression_r2"],
            "validation_regression_icc_2_1": metrics["regression_icc_2_1"],
        })
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
        print(json.dumps(metrics, indent=2))

        validation_r2 = metrics["regression_r2"]
        improved = validation_r2 > best_validation_r2 + args.early_stopping_min_delta
        if improved:
            best_validation_r2 = validation_r2
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint = {
            "epoch": epoch + 1,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scheduler_config": scheduler_config,
            "metrics": metrics,
            "history": history,
            "best_validation_r2": best_validation_r2,
            "epochs_without_improvement": epochs_without_improvement,
            "regression_standardization": regression_standardization,
            "classification_weighting": classification_weighting,
            "audio_augmentation": augmentation_config,
            "training_arguments": vars(args),
            "gradient_diagnostics": gradient_rows,
        }
        if improved:
            checkpoint_path = args.output_dir / "best_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved {checkpoint_path}")
            print(f"New best recording-level regression R-squared: {best_validation_r2:.6f}")
        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping after {args.early_stopping_patience} validation "
                "events without recording-level regression R-squared improvement"
            )
            break


if __name__ == "__main__":
    main()
