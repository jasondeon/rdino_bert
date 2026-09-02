from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import MultimodalCollator, MultimodalDataset
from model import BertRdinoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT + RDINO on a CSV manifest")
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--rdino-yaml", type=Path, default=Path("assets/rdino.yaml"))
    parser.add_argument("--rdino-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--text-model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--window-seconds", type=float, default=55.0)
    parser.add_argument("--stride-seconds", type=float, default=10.0)
    parser.add_argument("--include-empty-text", action="store_true")
    parser.add_argument("--max-text-tokens", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
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


def evaluate(model, loader, device, output_path: Path) -> dict[str, float]:
    model.eval()
    class_truth, class_predictions = [], []
    regression_truth, regression_predictions = [], []
    paths, subjects, window_starts, window_ends = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            tokens, waveforms, class_labels, regression_labels = move_batch(batch, device)
            class_logits, regression_output = model(tokens, waveforms)
            class_truth.extend(class_labels.cpu().tolist())
            class_predictions.extend(class_logits.argmax(dim=1).cpu().tolist())
            regression_truth.extend(regression_labels.cpu().tolist())
            regression_predictions.extend(regression_output.cpu().tolist())
            paths.extend(batch["audio_paths"])
            subjects.extend(batch["subject_ids"])
            window_starts.extend(batch["window_starts"])
            window_ends.extend(batch["window_ends"])

    correlation = spearmanr(regression_truth, regression_predictions).statistic
    metrics = {
        "accuracy": float(accuracy_score(class_truth, class_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(class_truth, class_predictions)),
        "macro_f1": float(f1_score(class_truth, class_predictions, average="macro")),
        "regression_mse": float(mean_squared_error(regression_truth, regression_predictions)),
        "regression_spearman": float(correlation),
    }
    pd.DataFrame({
        "audio_path": paths,
        "subject_id": subjects,
        "window_start": window_starts,
        "window_end": window_ends,
        "class_truth": class_truth,
        "class_prediction": class_predictions,
        "regression_truth": regression_truth,
        "regression_prediction": regression_predictions,
    }).to_csv(output_path, index=False)
    return metrics


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
    if len(train_data) < 2:
        raise ValueError("Training requires at least two examples because the fusion uses BatchNorm")
    print(
        f"Expanded {len(train_data.recordings)} training recordings into "
        f"{len(train_data)} windows (skipped {train_data.skipped_short_recordings} short); "
        f"{len(validation_data.recordings)} validation recordings into "
        f"{len(validation_data)} windows (skipped "
        f"{validation_data.skipped_short_recordings} short)"
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=len(train_data) % args.batch_size == 1,
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
        "fusion_dim": 128,
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
    regression_loss = torch.nn.MSELoss()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_number, batch in enumerate(train_loader, start=1):
            tokens, waveforms, class_labels, regression_labels = move_batch(batch, device)
            class_logits, regression_output = model(tokens, waveforms)
            loss = (
                args.classification_weight * classification_loss(class_logits, class_labels)
                + args.regression_weight * regression_loss(regression_output, regression_labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if batch_number == 1 or batch_number % 10 == 0:
                print(f"epoch={epoch + 1} batch={batch_number} loss={loss.item():.6f}")

        prediction_path = args.output_dir / f"predictions_epoch_{epoch + 1}.csv"
        metrics = evaluate(model, validation_loader, device, prediction_path)
        checkpoint_path = args.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "training_arguments": vars(args),
        }, checkpoint_path)
        print(f"epoch={epoch + 1} mean_train_loss={np.mean(losses):.6f}")
        print(json.dumps(metrics, indent=2))
        print(f"Saved {checkpoint_path}")


if __name__ == "__main__":
    main()
