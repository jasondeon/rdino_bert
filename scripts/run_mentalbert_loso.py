"""Standalone, resumable MentalBERT LOSO experiment on a frozen bundle.

Use --dry-run for CPU-only data/tokenizer preflight. A full run performs four
inner study fits and one final refit per outer study, sequentially on one device.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from clinical_evaluation import regression_metrics, stable_id
from mentalbert_evaluation import (
    MentalBertRegressor, adaptation_state, aggregate_chunks, collate_recordings,
    restore_adaptation, sample_recording_windows, select_epoch,
    target_standardization, tokenize_windows, windows_by_recording,
)
from scripts.build_evaluation_bundle import digest, write_csv
from scripts.score_evaluation_bundle import evaluate, read_csv, verify_bundle


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_checkpoint(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def guard_configuration(path, configuration):
    if path.exists():
        if json.loads(path.read_text()) != configuration:
            raise ValueError(f"Configuration changed; use a new output directory: {path}")
    else:
        atomic_json(path, configuration)


def make_model(config, device, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return MentalBertRegressor(
        config["model_name"], config["revision"], config["lora_rank"],
        config["lora_alpha"], config["normalization"],
        local_files_only=config["local_files_only"], weights_revision=config["weights_revision"],
    ).to(device)


def autocast_context(device, precision):
    if precision == "bfloat16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def predict_windows(model, windows, tokenizer, standardization, device, batch_size, precision):
    model.eval()
    flattened = [(window.window_id, chunk, weight)
                 for window in windows for chunk, weight in zip(window.chunks, window.chunk_weights)]
    result = {window.window_id: 0.0 for window in windows}
    for offset in range(0, len(flattened), batch_size):
        batch = flattened[offset:offset + batch_size]
        tokens = tokenizer.pad([chunk for _, chunk, _ in batch], padding=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with autocast_context(device, precision):
            values = model(tokens)
        values = values.float().cpu().numpy() * standardization["std"] + standardization["mean"]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite inference predictions")
        for (window_id, _, weight), value in zip(batch, values):
            result[window_id] += weight * float(value)
    return result


def recording_predictions(records, grouped, window_predictions):
    result = []
    for row in records:
        windows = grouped[row["recording_id"]]
        total = sum(window.duration for window in windows)
        result.append(sum(window.duration * window_predictions[window.window_id] for window in windows) / total)
    return result


def validate_subjects(train, validation, test):
    sets = [{r["subject_id"] for r in rows} for rows in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Subject leakage between train, validation, or test")


def fit_stage(directory, train, validation, grouped, tokenizer, config, device, epochs, seed):
    directory.mkdir(parents=True, exist_ok=True)
    specification = {
        "experiment": fingerprint(config), "epochs": epochs, "seed": seed,
        "train_ids": [r["recording_id"] for r in train],
        "validation_ids": [r["recording_id"] for r in validation],
    }
    guard_configuration(directory / "stage_config.json", specification)
    signature = fingerprint(specification)
    checkpoint_path = directory / "last_checkpoint.pt"
    complete_path = directory / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if complete["signature"] != signature or digest(checkpoint_path) != complete["checkpoint_sha256"]:
            raise ValueError(f"Completed stage integrity mismatch: {directory}")
        if digest(directory / "history.json") != complete["history_sha256"]:
            raise ValueError("Completed stage history changed")
        print(f"Skipping completed stage: {directory}", flush=True)
        return json.loads((directory / "history.json").read_text())
    standardization = target_standardization(train)
    model = make_model(config, device, seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    history, start_epoch = [], 1
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature or checkpoint["standardization"] != standardization:
            raise ValueError("Resume checkpoint does not match this stage")
        restore_adaptation(model, checkpoint["adaptation"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        del checkpoint
        print(f"Resuming {directory} at epoch {start_epoch}", flush=True)
    else:
        print(f"Starting {directory}: {len(train)} training recordings, {sum(p.numel() for p in parameters):,} trainable parameters", flush=True)
    validation_windows = [w for r in validation for w in grouped[r["recording_id"]]]
    for epoch in range(start_epoch, epochs + 1):
        started = time.monotonic()
        model.train()
        sampled = sample_recording_windows(train, grouped, config["windows_per_recording"], seed, epoch)
        batch_size = config["recordings_per_batch"]
        total_loss, total_recordings = 0.0, 0
        for offset in range(0, len(sampled), batch_size):
            examples = sampled[offset:offset + batch_size]
            tokens, owners, weights, targets = collate_recordings(examples, tokenizer, standardization)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            owners, weights, targets = owners.to(device), weights.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config["precision"]):
                chunk_predictions = model(tokens)
                prediction = aggregate_chunks(chunk_predictions.float(), owners, weights, len(examples))
                loss = torch.nn.functional.mse_loss(prediction, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss: {directory}, epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(examples)
            total_recordings += len(examples)
        training_mse = total_loss / total_recordings
        if validation and (epoch % config["validation_interval"] == 0 or epoch == epochs):
            window_prediction = predict_windows(
                model, validation_windows, tokenizer, standardization, device,
                config["inference_batch_size"], config["precision"],
            )
            predictions = recording_predictions(validation, grouped, window_prediction)
            metrics = regression_metrics([float(r["regression_label"]) for r in validation], predictions)
            history.append({"epoch": epoch, "training_mse_standardized": training_mse, **metrics})
            print(f"{directory.name} epoch {epoch}/{epochs}: train MSE={training_mse:.4f}, validation RMSE={metrics['rmse']:.3f}, R2={metrics['r2']}", flush=True)
        else:
            print(f"{directory.name} epoch {epoch}/{epochs}: train MSE={training_mse:.4f}, seconds={time.monotonic()-started:.1f}", flush=True)
        atomic_checkpoint(checkpoint_path, {
            "signature": signature, "epoch": epoch,
            "adaptation": adaptation_state(model), "optimizer": optimizer.state_dict(),
            "standardization": standardization, "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "model_config": config,
        })
        atomic_json(directory / "history.json", history)
    # Also restore history.json if a process was interrupted between checkpoint
    # replacement and its human-readable history export.
    atomic_json(directory / "history.json", history)
    atomic_json(complete_path, {
        "signature": signature, "epochs": epochs,
        "checkpoint_sha256": digest(checkpoint_path),
        "history_sha256": digest(directory / "history.json"),
    })
    del model, optimizer, parameters
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return history


def run_outer(study, bundle, output, grouped, tokenizer, config, device):
    fold = bundle / "loso" / study
    directory = output / study
    directory.mkdir(exist_ok=True)
    completed = directory / "complete.json"
    predictions_path = directory / "window_predictions.csv"
    if completed.exists():
        data = json.loads(completed.read_text())
        if data["experiment"] != fingerprint(config) or data["predictions_sha256"] != digest(predictions_path):
            raise ValueError("Completed outer-fold integrity mismatch")
        print(f"Skipping completed outer fold: {study}", flush=True)
        return
    train, test = read_csv(fold / "train.csv"), read_csv(fold / "test.csv")
    histories = []
    for inner in sorted((fold / "inner").iterdir()):
        fit, validation = read_csv(inner / "train.csv"), read_csv(inner / "validation.csv")
        validate_subjects(fit, validation, test)
        if {r["recording_id"] for r in fit + validation} != {r["recording_id"] for r in train}:
            raise ValueError("Inner partitions do not match outer training cohort")
        seed = (config["seed"] + int(stable_id(f"{study}/inner/{inner.name}")[:8], 16)) % (2**31)
        histories.append(fit_stage(directory / "inner" / inner.name, fit, validation, grouped,
                                   tokenizer, config, device, config["max_epochs"], seed))
    epoch, curve = select_epoch(histories)
    atomic_json(directory / "epoch_selection.json", {
        "selected_epoch": epoch, "selection": "minimum equal-study inner validation RMSE",
        "curve": curve, "selected_at_max_epochs": epoch == config["max_epochs"],
    })
    print(f"Outer {study}: selected {epoch} epochs from inner studies", flush=True)
    seed = (config["seed"] + int(stable_id(f"{study}/final")[:8], 16)) % (2**31)
    validate_subjects(train, [], test)
    fit_stage(directory / "final", train, [], grouped, tokenizer, config, device, epoch, seed)
    checkpoint = torch.load(directory / "final" / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    model = make_model(config, device, seed)
    restore_adaptation(model, checkpoint["adaptation"])
    windows = [w for r in test for w in grouped[r["recording_id"]]]
    predictions = predict_windows(model, windows, tokenizer, checkpoint["standardization"], device,
                                  config["inference_batch_size"], config["precision"])
    # No clipping or recalibration with held-out labels.
    rows = [{"window_id": w.window_id, "test_study": study, "prediction": predictions[w.window_id]} for w in windows]
    write_csv(predictions_path, rows)
    values = recording_predictions(test, grouped, predictions)
    write_csv(directory / "recording_predictions.csv", [
        {"recording_id": r["recording_id"], "test_study": study, "prediction": value}
        for r, value in zip(test, values)
    ])
    atomic_json(directory / "test_metrics.json", regression_metrics([float(r["regression_label"]) for r in test], values))
    atomic_json(completed, {"experiment": fingerprint(config), "selected_epoch": epoch,
                            "predictions_sha256": digest(predictions_path)})
    del checkpoint, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def preflight(args):
    verify_bundle(args.bundle)
    bundle_config = json.loads((args.bundle / "config.json").read_text())
    if bundle_config.get("max_gap_seconds") != 60.0:
        raise ValueError("This experiment expects a bundle with the agreed 60-second maximum gap")
    local_only = not args.allow_download
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=local_only)
    revision = model_config._commit_hash
    if revision is None and not Path(args.model).is_dir():
        raise ValueError("Could not resolve a fixed model revision")
    model_path = Path(args.model)
    # MentalBERT's cached safetensors conversion is published on refs/pr/5;
    # its config/tokenizer remain on the original repository revision. Pin both.
    weights_ref = args.weights_revision or ("refs/pr/5" if args.model == "mental/mental-bert-base-uncased" else revision)
    if model_path.is_dir():
        weights = model_path / "model.safetensors"
        index = model_path / "model.safetensors.index.json"
        weights_revision = None
    else:
        weights = try_to_load_from_cache(args.model, "model.safetensors", revision=weights_ref)
        index = try_to_load_from_cache(args.model, "model.safetensors.index.json", revision=weights_ref)
        if not local_only and not isinstance(weights, str) and not isinstance(index, str):
            weights = hf_hub_download(args.model, "model.safetensors", revision=weights_ref)
        resolved_file = weights if isinstance(weights, str) else index
        if not isinstance(resolved_file, str):
            raise ValueError("Model safetensors are not cached; rerun with --allow-download")
        weights_revision = Path(resolved_file).parent.name
    if not (isinstance(weights, (str, Path)) and Path(weights).is_file()):
        if not (isinstance(index, (str, Path)) and Path(index).is_file()):
            raise ValueError("Model safetensors are not cached; rerun with --allow-download")
        for shard in set(json.loads(Path(index).read_text())["weight_map"].values()):
            if not (Path(index).parent / shard).is_file():
                raise ValueError(f"Missing cached model shard: {shard}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, local_files_only=local_only)
    if args.max_tokens > model_config.max_position_embeddings:
        raise ValueError("Token limit exceeds model positional capacity")
    windows = tokenize_windows(read_csv(args.bundle / "windows.csv"), tokenizer, args.max_tokens)
    grouped = windows_by_recording(windows)
    records = read_csv(args.bundle / "recordings.csv")
    if set(grouped) != {r["recording_id"] for r in records}:
        raise ValueError("Text view must cover exactly the common recording cohort")
    studies = sorted(r.name for r in (args.bundle / "loso").iterdir())
    if args.study and not set(args.study) <= set(studies):
        raise ValueError(f"Unknown study; expected one of {studies}")
    for study in studies:
        fold = args.bundle / "loso" / study
        test = read_csv(fold / "test.csv")
        validate_subjects(read_csv(fold / "train.csv"), [], test)
        for inner in (fold / "inner").iterdir():
            validate_subjects(read_csv(inner / "train.csv"), read_csv(inner / "validation.csv"), test)
    root = Path(__file__).resolve().parents[1]
    config = {
        "bundle_checksums_sha256": digest(args.bundle / "checksums.json"),
        "model_name": args.model, "revision": revision, "weights_revision": weights_revision, "local_files_only": local_only,
        "local_model_sha256": {str(p.relative_to(Path(args.model))): digest(p) for p in sorted(Path(args.model).rglob("*")) if p.is_file()} if Path(args.model).is_dir() else None,
        "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
        "normalization": args.normalization, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "max_epochs": args.max_epochs,
        "validation_interval": args.validation_interval,
        "recordings_per_batch": args.recordings_per_batch,
        "windows_per_recording": args.windows_per_recording,
        "inference_batch_size": args.inference_batch_size, "max_tokens": args.max_tokens,
        "gradient_clip": 1.0, "precision": args.precision, "seed": args.seed,
        "objective": "recording-level standardized MADRS MSE; no classification loss",
        "sampling": "four duration-proportional draws by default; each recording once per epoch",
        "optimizer": "AdamW, constant learning rate, no scheduler",
        "selection": "equal-study inner RMSE chooses epoch; final refit uses outer train only",
        "code_sha256": {str(p.relative_to(root)): digest(p) for p in [Path(__file__).resolve(), root / "mentalbert_evaluation.py", root / "clinical_evaluation.py", root / "scripts/score_evaluation_bundle.py"]},
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
    }
    report = {
        "recordings": len(records), "subjects": len({r["subject_id"] for r in records}),
        "studies": studies, "text_windows": len(windows),
        "token_chunks": sum(len(w.chunks) for w in windows.values()),
        "windows_requiring_token_split": sum(len(w.chunks) > 1 for w in windows.values()),
        "max_encoded_length": max(len(c["input_ids"]) for w in windows.values() for c in w.chunks),
        "outer_fits": len(studies), "inner_fits": sum(len(list((args.bundle / "loso" / s / "inner").iterdir())) for s in studies),
        "revision": revision, "weights_revision": weights_revision,
        "training_started": False,
    }
    return config, report, tokenizer, grouped, studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("outputs/evaluation/20260913_v3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mentalbert-loso-20260913"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--study", action="append", help="Optional outer study; repeat to select several. Default: all five")
    parser.add_argument("--model", default="mental/mental-bert-base-uncased")
    parser.add_argument("--weights-revision", help="Override model-weight revision; MentalBERT defaults to its safetensors conversion refs/pr/5")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1.1962402885307259e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--normalization", choices=("batchnorm", "layernorm"), default="batchnorm")
    parser.add_argument("--recordings-per-batch", type=int, default=2)
    parser.add_argument("--windows-per-recording", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    for name in ("max_epochs", "validation_interval", "recordings_per_batch", "windows_per_recording", "inference_batch_size", "max_tokens", "lora_rank", "lora_alpha"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.normalization == "batchnorm" and args.windows_per_recording < 2:
        parser.error("BatchNorm requires at least two sampled windows, including for the final one-recording batch")
    if args.seed < 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Invalid seed, learning rate, or weight decay")
    args.bundle = args.bundle.resolve()
    args.output = args.output.resolve()
    if args.output == args.bundle or args.bundle in args.output.parents:
        parser.error("Experiment outputs must be outside the immutable bundle")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process is using this experiment output")
        config, report, tokenizer, grouped, studies = preflight(args)
        guard_configuration(args.output / "experiment_config.json", config)
        atomic_json(args.output / "preflight.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if args.dry_run:
            print("CPU preflight complete. No model weights loaded or training started.")
            return
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu only for small tests")
        if args.precision == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16")
        torch.backends.cudnn.benchmark = False
        tokenizer.save_pretrained(args.output / "tokenizer")
        for study in args.study or studies:
            run_outer(study, args.bundle, args.output, grouped, tokenizer, config, device)
        if all((args.output / study / "complete.json").exists() for study in studies):
            for study in studies:
                completion = json.loads((args.output / study / "complete.json").read_text())
                if completion["experiment"] != fingerprint(config) or completion["predictions_sha256"] != digest(args.output / study / "window_predictions.csv"):
                    raise ValueError(f"Outer fold integrity mismatch before combined scoring: {study}")
            rows = [row for study in studies for row in read_csv(args.output / study / "window_predictions.csv")]
            write_csv(args.output / "loso_window_predictions.csv", rows)
            metrics, recording_rows = evaluate(args.bundle, rows, "window", "text")
            metrics["experiment_config_sha256"] = digest(args.output / "experiment_config.json")
            atomic_json(args.output / "metrics.json", metrics)
            write_csv(args.output / "loso_recording_predictions.csv", recording_rows)
            print(json.dumps({"macro_study": metrics["macro_study"], "pooled": metrics["pooled"]}, indent=2))
        else:
            print("Selected folds complete. Rerun without --study to finish all folds and obtain the combined report.")


if __name__ == "__main__":
    main()
