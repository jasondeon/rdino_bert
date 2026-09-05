from __future__ import annotations

import argparse
import json
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
from torch.utils.data import DataLoader, Sampler
from transformers import AutoTokenizer

from dataset import MultimodalCollator, MultimodalDataset
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT + RDINO on a CSV manifest")
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--rdino-yaml", type=Path, default=Path("assets/rdino.yaml"))
    parser.add_argument("--rdino-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--text-model", default="mental/mental-bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--train-sampling",
        choices=("recording", "window"),
        default="recording",
        help="Sample grouped windows per recording (default) or every window each epoch",
    )
    parser.add_argument(
        "--train-windows-per-recording",
        type=int,
        default=4,
        help="Windows averaged for each recording-level training loss (default: 4)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many epochs without recording-level R-squared improvement",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--window-seconds", type=float, default=55.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--include-empty-text", action="store_true")
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument(
        "--standardize-regression-labels",
        action="store_true",
        help="Fit mean/std on training-recording labels and train in standardized units",
    )
    parser.add_argument("--seed", type=int, default=40)
    return parser.parse_args()


def move_batch(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = {name: value.to(device, non_blocking=True) for name, value in batch["tokens"].items()}
    return (
        tokens,
        batch["waveforms"].to(device, non_blocking=True),
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
    classification_loss = float(
        -np.log(mean_probabilities[np.arange(len(class_truth)), class_truth].clip(1e-12)).mean()
    )
    regression_loss = float(mean_squared_error(standardized_truth, standardized_predictions))
    regression_rmse = float(np.sqrt(regression_loss))
    regression_rmse_original_scale = float(
        np.sqrt(mean_squared_error(regression_truth, regression_predictions))
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
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    collator = MultimodalCollator(tokenizer, args.max_text_tokens)
    common = dict(
        sample_rate=16_000,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        num_classes=args.num_classes,
        include_empty_text=args.include_empty_text,
    )
    train_data = MultimodalDataset(args.train_manifest, **common)
    validation_data = MultimodalDataset(args.validation_manifest, **common)
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
    else:
        train_sampler = None
        recordings_per_batch = None
    samples_per_epoch = len(train_sampler) if train_sampler is not None else len(train_data)
    print(
        f"Training sampling: {args.train_sampling}; "
        f"{samples_per_epoch} windows per epoch"
    )
    if recording_level_training:
        print(
            f"Recording-level loss: {args.train_windows_per_recording} windows per "
            f"recording; {recordings_per_batch} recordings per optimizer batch"
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
        "lora_rank": 2,
        "lora_alpha": 16,
        "sample_rate": 16_000,
    }
    model = BertRdinoModel(**model_config).to(device)
    trainable, total = model.trainable_parameter_counts()
    print(f"Device: {device}; trainable parameters: {trainable:,}/{total:,}")

    classification_loss = torch.nn.CrossEntropyLoss()
    recording_classification_loss = torch.nn.NLLLoss()
    regression_loss = torch.nn.MSELoss()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1")
    history: list[dict[str, float | int]] = []
    best_validation_r2 = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        total_losses, classification_losses, regression_losses = [], [], []
        for batch_number, batch in enumerate(train_loader, start=1):
            tokens, waveforms, class_labels, regression_labels = move_batch(batch, device)
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
                )
            else:
                batch_classification_loss = classification_loss(
                    class_logits, class_labels
                )
            regression_targets = _standardize_regression(
                regression_labels, regression_standardization
            )
            batch_regression_loss = regression_loss(regression_output, regression_targets)
            loss = (
                args.classification_weight * batch_classification_loss
                + args.regression_weight * batch_regression_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_losses.append(loss.item())
            classification_losses.append(batch_classification_loss.item())
            regression_losses.append(batch_regression_loss.item())
            if batch_number == 1 or batch_number % 10 == 0:
                print(f"epoch={epoch + 1} batch={batch_number} loss={loss.item():.6f}")

        prediction_path = args.output_dir / f"predictions_epoch_{epoch + 1}.csv"
        metrics = evaluate(
            model,
            validation_loader,
            device,
            prediction_path,
            regression_standardization,
            args.classification_weight,
            args.regression_weight,
        )
        history.append({
            "epoch": epoch + 1,
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
        })
        pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
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
            "metrics": metrics,
            "history": history,
            "best_validation_r2": best_validation_r2,
            "epochs_without_improvement": epochs_without_improvement,
            "regression_standardization": regression_standardization,
            "training_arguments": vars(args),
        }
        if improved:
            checkpoint_path = args.output_dir / "best_checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
        print(f"epoch={epoch + 1} mean_train_loss={np.mean(total_losses):.6f}")
        print(json.dumps(metrics, indent=2))
        if improved:
            print(f"Saved {checkpoint_path}")
            print(f"New best recording-level regression R-squared: {best_validation_r2:.6f}")
        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping after {args.early_stopping_patience} epochs without "
                "recording-level regression R-squared improvement"
            )
            break


if __name__ == "__main__":
    main()
