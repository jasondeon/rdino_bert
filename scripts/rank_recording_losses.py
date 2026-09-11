from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MultimodalCollator, MultimodalDataset
from model import BertRdinoModel
from train import evaluate


RANK_COLUMNS = {
    "combined": "combined_loss",
    "regression": "regression_loss",
    "absolute-error": "regression_absolute_error",
    "classification": "classification_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank full recordings by loss from a saved training checkpoint"
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Manifest to audit; defaults to the validation manifest saved in "
            "the checkpoint"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination CSV; defaults to recording_loss_audit.csv beside "
            "the checkpoint"
        ),
    )
    parser.add_argument(
        "--rank-by",
        choices=tuple(RANK_COLUMNS),
        default="combined",
        help="Loss component used for descending rank (default: combined)",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cpu, cuda, or cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and show checkpoint settings without running inference",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {name}")
    return device


def checkpoint_settings(
    checkpoint: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    training_arguments = checkpoint.get("training_arguments", {})
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise KeyError("Checkpoint does not contain model_config")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict")

    saved_manifest = training_arguments.get("validation_manifest")
    if args.manifest is None and saved_manifest is None:
        raise ValueError(
            "Checkpoint has no validation manifest; provide --manifest explicitly"
        )
    manifest = resolve_path(
        args.manifest if args.manifest is not None else Path(saved_manifest)
    )
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    checkpoint_path = resolve_path(args.checkpoint)
    output = (
        resolve_path(args.output)
        if args.output is not None
        else checkpoint_path.parent / "recording_loss_audit.csv"
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(training_arguments.get("batch_size", 8))
    )
    workers = (
        args.workers
        if args.workers is not None
        else int(training_arguments.get("workers", 0))
    )
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    return {
        "checkpoint_path": checkpoint_path,
        "manifest": manifest,
        "output": output,
        "batch_size": batch_size,
        "workers": workers,
        "training_arguments": training_arguments,
        "model_config": model_config,
        "regression_standardization": checkpoint.get(
            "regression_standardization",
            {"enabled": False, "mean": 0.0, "std": 1.0},
        ),
        "classification_weighting": checkpoint.get(
            "classification_weighting",
            {"method": "none", "weights": [1.0] * int(model_config["num_classes"])},
        ),
    }


def add_recording_losses(
    frame: pd.DataFrame,
    dataset: MultimodalDataset,
    class_weights: list[float],
    classification_weight: float,
    regression_weight: float,
) -> pd.DataFrame:
    result = frame.copy()
    class_truth = result["class_truth"].to_numpy(dtype=int)
    probability_columns = [
        f"class_probability_{index}" for index in range(len(class_weights))
    ]
    missing = set(probability_columns).difference(result.columns)
    if missing:
        raise ValueError(f"Prediction output is missing columns: {sorted(missing)}")
    probabilities = result[probability_columns].to_numpy(dtype=float)
    true_class_probability = probabilities[
        np.arange(len(result)), class_truth
    ].clip(1e-12)
    classification_nll = -np.log(true_class_probability)
    sample_class_weights = np.asarray(class_weights, dtype=float)[class_truth]

    standardized_error = (
        result["regression_prediction_standardized"].to_numpy(dtype=float)
        - result["regression_truth_standardized"].to_numpy(dtype=float)
    )
    original_error = (
        result["regression_prediction"].to_numpy(dtype=float)
        - result["regression_truth"].to_numpy(dtype=float)
    )
    result["class_truth_probability"] = true_class_probability
    result["classification_nll"] = classification_nll
    result["classification_loss"] = classification_nll * sample_class_weights
    result["regression_error"] = original_error
    result["regression_absolute_error"] = np.abs(original_error)
    result["regression_squared_error_original_scale"] = np.square(original_error)
    result["regression_loss"] = np.square(standardized_error)
    result["combined_loss"] = (
        classification_weight * result["classification_loss"]
        + regression_weight * result["regression_loss"]
    )

    metadata = pd.DataFrame(
        {
            "recording_index": np.arange(len(dataset.recordings)),
            "word_timestamps_path": [
                str(recording.word_timestamps_path)
                for recording in dataset.recordings
            ],
            "diarization_path": [
                str(recording.diarization_path) for recording in dataset.recordings
            ],
        }
    )
    return result.merge(
        metadata,
        on="recording_index",
        how="left",
        validate="one_to_one",
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    settings = checkpoint_settings(checkpoint, args)
    device = resolve_device(args.device)
    training_arguments = settings["training_arguments"]
    model_config = settings["model_config"]
    summary = {
        "checkpoint": str(settings["checkpoint_path"]),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest": str(settings["manifest"]),
        "output": str(settings["output"]),
        "device": str(device),
        "modality": model_config["modality"],
        "window_seconds": training_arguments.get("window_seconds", 30.0),
        "stride_seconds": training_arguments.get("stride_seconds", 20.0),
        "eligibility_window_seconds": training_arguments.get(
            "eligibility_window_seconds"
        ),
        "rank_by": args.rank_by,
        "top_k": args.top_k,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("Dry run complete; inference was not launched.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_config["text_model_name"])
    collator = MultimodalCollator(
        tokenizer,
        int(training_arguments.get("max_text_tokens", 512)),
    )
    dataset = MultimodalDataset(
        settings["manifest"],
        sample_rate=int(model_config.get("sample_rate", 16_000)),
        window_seconds=float(training_arguments.get("window_seconds", 30.0)),
        stride_seconds=float(training_arguments.get("stride_seconds", 20.0)),
        num_classes=int(model_config["num_classes"]),
        include_empty_text=bool(
            training_arguments.get("include_empty_text", False)
        ),
        load_audio=model_config["modality"] != "text",
        eligibility_window_seconds=training_arguments.get(
            "eligibility_window_seconds"
        ),
        speaker_gap_policy=training_arguments.get(
            "speaker_gap_policy", "concatenate"
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=False,
        num_workers=settings["workers"],
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    model = BertRdinoModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)

    class_weights = [
        float(value)
        for value in settings["classification_weighting"].get(
            "weights", [1.0] * int(model_config["num_classes"])
        )
    ]
    classification_weight = float(
        training_arguments.get("classification_weight", 1.0)
    )
    regression_weight = float(training_arguments.get("regression_weight", 1.0))
    with tempfile.TemporaryDirectory(prefix="rdino_recording_audit_") as directory:
        window_output = Path(directory) / "predictions.csv"
        metrics = evaluate(
            model,
            loader,
            device,
            window_output,
            settings["regression_standardization"],
            class_weights=class_weights,
            classification_weight=classification_weight,
            regression_weight=regression_weight,
        )
        recording_predictions = pd.read_csv(
            window_output.with_name(f"recording_{window_output.name}")
        )

    ranked = add_recording_losses(
        recording_predictions,
        dataset,
        class_weights,
        classification_weight,
        regression_weight,
    ).sort_values(RANK_COLUMNS[args.rank_by], ascending=False, kind="stable")
    ranked.insert(0, "loss_rank", np.arange(1, len(ranked) + 1))
    output_path: Path = settings["output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_path, index=False)

    display_columns = [
        "loss_rank",
        "recording_index",
        "subject_id",
        "audio_path",
        "regression_truth",
        "regression_prediction",
        "regression_absolute_error",
        "regression_loss",
        "classification_loss",
        "combined_loss",
        "window_count",
    ]
    print("\nValidation metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nHighest-loss recordings ranked by {RANK_COLUMNS[args.rank_by]}:")
    with pd.option_context("display.max_colwidth", 80, "display.width", 220):
        print(ranked[display_columns].head(args.top_k).to_string(index=False))
    print(f"\nSaved all {len(ranked)} ranked recordings to {output_path}")


if __name__ == "__main__":
    main()
