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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_metrics import intraclass_correlation_2_1


class WavLMLayerMixer(nn.Module):
    def __init__(self, layer_count: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.mean_layer_logits = nn.Parameter(torch.zeros(layer_count))
        self.std_layer_logits = nn.Parameter(torch.zeros(layer_count))
        self.normalization = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.regression = nn.Linear(hidden_size * 2, 1)

    def forward(
        self, layer_means: torch.Tensor, layer_stds: torch.Tensor
    ) -> torch.Tensor:
        mean_weights = self.mean_layer_logits.softmax(dim=0)
        std_weights = self.std_layer_logits.softmax(dim=0)
        mixed_mean = torch.einsum("l,bld->bd", mean_weights, layer_means)
        mixed_std = torch.einsum("l,bld->bd", std_weights, layer_stds)
        features = torch.cat((mixed_mean, mixed_std), dim=-1)
        return self.regression(self.dropout(self.normalization(features))).squeeze(-1)

    def layer_weights(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.mean_layer_logits.softmax(0).detach().cpu().numpy(),
            self.std_layer_logits.softmax(0).detach().cpu().numpy(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight supervised mixer over cached WavLM layers"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/wavlm-base-plus-layer-audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wavlm-base-plus-layer-mixer"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=(40, 41, 42))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--minimum-delta", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cpu, cuda, or cuda:0",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {name}")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(
    input_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    window_means = np.load(input_dir / f"{split}_layer_means.npy", mmap_mode="r")
    window_stds = np.load(input_dir / f"{split}_layer_stds.npy", mmap_mode="r")
    windows = pd.read_csv(
        input_dir / f"{split}_windows.csv", dtype={"subject_id": str}
    )
    if (
        window_means.ndim != 3
        or window_stds.shape != window_means.shape
        or len(window_means) != len(windows)
    ):
        raise ValueError(f"Invalid or misaligned {split} WavLM cache")
    if not np.array_equal(
        windows["embedding_index"].to_numpy(), np.arange(len(windows))
    ):
        raise ValueError(f"Nonsequential {split} embedding indices")

    recording_means = []
    recording_stds = []
    rows = []
    for recording_index, group in windows.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(dtype=int)
        labels = group["regression_label"].to_numpy(dtype=float)
        if not np.allclose(labels, labels[0]):
            raise ValueError(f"Labels differ within recording {recording_index}")
        recording_means.append(
            np.asarray(window_means[indices], dtype=np.float32).mean(axis=0)
        )
        recording_stds.append(
            np.asarray(window_stds[indices], dtype=np.float32).mean(axis=0)
        )
        first = group.iloc[0]
        rows.append(
            {
                "recording_index": int(recording_index),
                "audio_path": str(first["audio_path"]),
                "subject_id": str(first["subject_id"]),
                "class_label": int(first["class_label"]),
                "regression_truth": float(labels[0]),
                "window_count": len(group),
            }
        )
    return np.stack(recording_means), np.stack(recording_stds), pd.DataFrame(rows)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth_std = float(np.std(truth, ddof=1))
    prediction_std = float(np.std(prediction, ddof=1))
    pearson = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if truth_std > 0 and prediction_std > 0
        else float("nan")
    )
    return {
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "icc_2_1": intraclass_correlation_2_1(truth, prediction),
        "pearson": pearson,
        "prediction_target_sd_ratio": prediction_std / truth_std,
        "mean_error": float(np.mean(prediction - truth)),
    }


def bootstrap_interval(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(truth), len(truth))
        if np.var(truth[indices]) > 0:
            values.append(r2_score(truth[indices], prediction[indices]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def paired_r2_difference(
    truth: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    observed = float(r2_score(truth, prediction) - r2_score(truth, reference))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(truth), len(truth))
        if np.var(truth[indices]) > 0:
            values.append(
                r2_score(truth[indices], prediction[indices])
                - r2_score(truth[indices], reference[indices])
            )
    low, high = np.percentile(values, [2.5, 97.5])
    return {"delta_r2": observed, "ci_low": float(low), "ci_high": float(high)}


def predict(
    model: WavLMLayerMixer,
    means: torch.Tensor,
    stds: torch.Tensor,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        standardized = model(means, stds)
    return standardized.cpu().numpy() * target_std + target_mean


def fit_with_early_stopping(
    means: torch.Tensor,
    stds: torch.Tensor,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
) -> tuple[WavLMLayerMixer, int, np.ndarray, list[dict[str, float]]]:
    set_seed(seed)
    model = WavLMLayerMixer(means.shape[1], means.shape[2], args.dropout).to(
        means.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    target_mean = float(targets[train_indices].mean())
    target_std = float(targets[train_indices].std(ddof=0))
    if target_std <= 0:
        raise ValueError("Training targets have zero variance")
    standardized_targets = torch.as_tensor(
        (targets - target_mean) / target_std,
        dtype=torch.float32,
        device=means.device,
    )
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=means.device)
    validation_tensor = torch.as_tensor(
        validation_indices, dtype=torch.long, device=means.device
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_indices), generator=generator)
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(permutation), args.batch_size):
            cpu_positions = permutation[start : start + args.batch_size]
            batch_indices = train_tensor[cpu_positions.to(train_tensor.device)]
            optimizer.zero_grad(set_to_none=True)
            output = model(means[batch_indices], stds[batch_indices])
            loss = nn.functional.mse_loss(output, standardized_targets[batch_indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_indices)
            total_count += len(batch_indices)
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                nn.functional.mse_loss(
                    model(means[validation_tensor], stds[validation_tensor]),
                    standardized_targets[validation_tensor],
                )
            )
        history.append(
            {
                "epoch": epoch,
                "train_standardized_mse": total_loss / total_count,
                "validation_standardized_mse": validation_loss,
            }
        )
        if validation_loss < best_loss - args.minimum_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("Early stopping did not save a model")
    model.load_state_dict(best_state)
    validation_prediction = predict(
        model,
        means[validation_tensor],
        stds[validation_tensor],
        target_mean,
        target_std,
    )
    return model, best_epoch, validation_prediction, history


def fit_full(
    means: torch.Tensor,
    stds: torch.Tensor,
    targets: np.ndarray,
    *,
    epochs: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[WavLMLayerMixer, float, float, list[dict[str, float]]]:
    set_seed(seed)
    model = WavLMLayerMixer(means.shape[1], means.shape[2], args.dropout).to(
        means.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    target_mean = float(targets.mean())
    target_std = float(targets.std(ddof=0))
    standardized_targets = torch.as_tensor(
        (targets - target_mean) / target_std,
        dtype=torch.float32,
        device=means.device,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(targets), generator=generator)
        total_loss = 0.0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(means.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(means[indices], stds[indices])
            loss = nn.functional.mse_loss(output, standardized_targets[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        history.append(
            {"epoch": epoch, "train_standardized_mse": total_loss / len(targets)}
        )
    return model, target_mean, target_std, history


def append_weights(
    rows: list[dict[str, Any]],
    model: WavLMLayerMixer,
    layer_names: list[str],
    *,
    seed: int,
    stage: str,
    fold: int | None,
) -> None:
    mean_weights, std_weights = model.layer_weights()
    for index, name in enumerate(layer_names):
        rows.append(
            {
                "seed": seed,
                "stage": stage,
                "fold": fold,
                "layer_index": index,
                "layer_name": name,
                "mean_weight": mean_weights[index],
                "std_weight": std_weights[index],
            }
        )


def plot_weights(weights: pd.DataFrame, output_path: Path) -> None:
    final = weights[weights["stage"] == "full"].copy()
    summary = final.groupby("layer_index").agg(
        mean_weight=("mean_weight", "mean"),
        mean_weight_sd=("mean_weight", "std"),
        std_weight=("std_weight", "mean"),
        std_weight_sd=("std_weight", "std"),
    )
    x = summary.index.to_numpy()
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.errorbar(
        x,
        summary["mean_weight"],
        yerr=summary["mean_weight_sd"].fillna(0),
        marker="o",
        capsize=3,
        label="Temporal mean mixture",
    )
    axis.errorbar(
        x,
        summary["std_weight"],
        yerr=summary["std_weight_sd"].fillna(0),
        marker="o",
        capsize=3,
        label="Temporal SD mixture",
    )
    axis.axhline(1 / len(summary), color="black", linestyle="--", alpha=0.5)
    axis.set(
        title="Learned WavLM layer weights across final seeds",
        xlabel="Hidden-state index",
        ylabel="Softmax weight",
        xticks=x,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.folds < 2 or not args.seeds:
        raise ValueError("At least two folds and one seed are required")
    if args.batch_size < 1 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Batch size, max epochs, and patience must be positive")
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0:
        raise ValueError("Invalid dropout or learning rate")
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extraction_config = json.loads(
        (input_dir / "extraction_config.json").read_text()
    )
    layer_names = list(extraction_config["layer_names"])
    device = resolve_device(args.device)
    configuration = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "folds": args.folds,
        "seeds": args.seeds,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "minimum_delta": args.minimum_delta,
        "gradient_clip": args.gradient_clip,
        "bootstrap_samples": args.bootstrap_samples,
        "selection": "median grouped-CV best epoch per seed",
        "external_validation_used_for_selection": False,
    }
    print(json.dumps(configuration, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; cached features were not loaded and no training ran.")
        return

    train_means_np, train_stds_np, train_metadata = load_split(input_dir, "train")
    validation_means_np, validation_stds_np, validation_metadata = load_split(
        input_dir, "validation"
    )
    if train_means_np.shape[1] != len(layer_names):
        raise ValueError("Cached layer count does not match extraction config")
    overlap = set(train_metadata["subject_id"]) & set(
        validation_metadata["subject_id"]
    )
    if overlap:
        raise ValueError(f"Subject leakage in cached data: {sorted(overlap)[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_means = torch.as_tensor(train_means_np, device=device)
    train_stds = torch.as_tensor(train_stds_np, device=device)
    validation_means = torch.as_tensor(validation_means_np, device=device)
    validation_stds = torch.as_tensor(validation_stds_np, device=device)
    train_targets = train_metadata["regression_truth"].to_numpy(dtype=float)
    validation_targets = validation_metadata["regression_truth"].to_numpy(dtype=float)
    groups = train_metadata["subject_id"].astype(str).to_numpy()
    splitter = list(GroupKFold(args.folds).split(train_targets, groups=groups))

    cv_rows = []
    history_rows = []
    weight_rows: list[dict[str, Any]] = []
    oof_by_seed = []
    validation_by_seed = []
    for seed in args.seeds:
        oof = np.full(len(train_targets), np.nan)
        best_epochs = []
        for fold, (train_indices, held_out_indices) in enumerate(splitter, start=1):
            model, best_epoch, fold_prediction, history = fit_with_early_stopping(
                train_means,
                train_stds,
                train_targets,
                train_indices,
                held_out_indices,
                args=args,
                seed=seed * 100 + fold,
            )
            oof[held_out_indices] = fold_prediction
            best_epochs.append(best_epoch)
            fold_metrics = metrics(train_targets[held_out_indices], fold_prediction)
            cv_rows.append(
                {"seed": seed, "fold": fold, "best_epoch": best_epoch, **fold_metrics}
            )
            for row in history:
                history_rows.append({"seed": seed, "fold": fold, **row})
            append_weights(
                weight_rows,
                model,
                layer_names,
                seed=seed,
                stage="cv",
                fold=fold,
            )
            print(
                f"seed={seed} fold={fold} best_epoch={best_epoch} "
                f"rmse={fold_metrics['rmse']:.4f}",
                flush=True,
            )
        if np.isnan(oof).any():
            raise RuntimeError("OOF predictions are incomplete")
        oof_by_seed.append(oof)
        final_epochs = max(1, int(round(float(np.median(best_epochs)))))
        model, target_mean, target_std, full_history = fit_full(
            train_means,
            train_stds,
            train_targets,
            epochs=final_epochs,
            args=args,
            seed=seed,
        )
        validation_prediction = predict(
            model, validation_means, validation_stds, target_mean, target_std
        )
        validation_by_seed.append(validation_prediction)
        append_weights(
            weight_rows,
            model,
            layer_names,
            seed=seed,
            stage="full",
            fold=None,
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "layer_names": layer_names,
                "hidden_size": train_means.shape[2],
                "dropout": args.dropout,
                "target_mean": target_mean,
                "target_std": target_std,
                "epochs": final_epochs,
                "seed": seed,
            },
            output_dir / f"model_seed_{seed}.pt",
        )
        for row in full_history:
            history_rows.append({"seed": seed, "fold": 0, **row})
        print(f"seed={seed} full_training_epochs={final_epochs}", flush=True)

    oof_matrix = np.stack(oof_by_seed, axis=1)
    validation_matrix = np.stack(validation_by_seed, axis=1)
    oof_ensemble = oof_matrix.mean(axis=1)
    validation_ensemble = validation_matrix.mean(axis=1)
    oof_metrics = metrics(train_targets, oof_ensemble)
    validation_metrics = metrics(validation_targets, validation_ensemble)
    validation_low, validation_high = bootstrap_interval(
        validation_targets,
        validation_ensemble,
        samples=args.bootstrap_samples,
        seed=args.seeds[0],
    )

    oof_frame = train_metadata.copy()
    for index, seed in enumerate(args.seeds):
        oof_frame[f"prediction_seed_{seed}"] = oof_matrix[:, index]
    oof_frame["regression_prediction"] = oof_ensemble
    oof_frame["regression_error"] = oof_ensemble - train_targets
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    validation_frame = validation_metadata.copy()
    for index, seed in enumerate(args.seeds):
        validation_frame[f"prediction_seed_{seed}"] = validation_matrix[:, index]
    validation_frame["regression_prediction"] = validation_ensemble
    validation_frame["regression_error"] = validation_ensemble - validation_targets
    validation_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / "cv_folds.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(output_dir / "layer_weights.csv", index=False)
    plot_weights(weights, output_dir / "layer_weights.png")

    baseline_prediction = np.full_like(validation_targets, train_targets.mean())
    comparisons = {
        "versus_training_mean": paired_r2_difference(
            validation_targets,
            validation_ensemble,
            baseline_prediction,
            samples=args.bootstrap_samples,
            seed=args.seeds[0] + 1000,
        )
    }
    fixed_path = input_dir / "selected_probe_predictions.csv"
    if fixed_path.exists():
        fixed = pd.read_csv(fixed_path, dtype={"subject_id": str})
        keys = ["recording_index", "audio_path", "subject_id"]
        joined = validation_frame[keys + ["regression_truth"]].merge(
            fixed[keys + ["regression_prediction"]],
            on=keys,
            validate="one_to_one",
        )
        if len(joined) != len(validation_frame):
            raise ValueError("Fixed-layer and mixer validation cohorts differ")
        comparisons["versus_fixed_cv_selected_layer"] = paired_r2_difference(
            joined["regression_truth"].to_numpy(float),
            validation_ensemble,
            joined["regression_prediction"].to_numpy(float),
            samples=args.bootstrap_samples,
            seed=args.seeds[0] + 2000,
        )

    results = {
        "configuration": configuration,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "oof_ensemble": oof_metrics,
        "validation_ensemble": {
            **validation_metrics,
            "r2_ci_low": validation_low,
            "r2_ci_high": validation_high,
        },
        "comparisons": comparisons,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )

    final_weights = weights[weights["stage"] == "full"]
    weight_summary = final_weights.groupby(["layer_index", "layer_name"])[
        ["mean_weight", "std_weight"]
    ].agg(["mean", "std"])
    lines = [
        "# WavLM lightweight layer mixer",
        "",
        "WavLM remained frozen. Separate softmax mixtures combine temporal means",
        "and temporal standard deviations from all hidden states. Their concatenation",
        "passes through LayerNorm, dropout, and a linear regression head.",
        "",
        "External validation did not control early stopping or any hyperparameter.",
        "Grouped training folds selected the epoch count independently for each seed.",
        "",
        "## Results",
        "",
        f"- Training out-of-fold ensemble: R² {oof_metrics['r2']:.4f}, RMSE "
        f"{oof_metrics['rmse']:.4f}, ICC {oof_metrics['icc_2_1']:.4f}, Pearson "
        f"{oof_metrics['pearson']:.4f}.",
        f"- External validation ensemble: R² {validation_metrics['r2']:.4f} "
        f"[{validation_low:.4f}, {validation_high:.4f}], RMSE "
        f"{validation_metrics['rmse']:.4f}, ICC {validation_metrics['icc_2_1']:.4f}, "
        f"Pearson {validation_metrics['pearson']:.4f}, prediction/target SD "
        f"{validation_metrics['prediction_target_sd_ratio']:.4f}.",
        "",
        "## Paired validation comparisons",
        "",
    ]
    for name, comparison in comparisons.items():
        lines.append(
            f"- {name}: ΔR² {comparison['delta_r2']:.4f} "
            f"[{comparison['ci_low']:.4f}, {comparison['ci_high']:.4f}]."
        )
    lines.extend(
        [
            "",
            "## Final layer weights across seeds",
            "",
            "| Layer | Mean-stream weight | SD-stream weight |",
            "| ---: | ---: | ---: |",
        ]
    )
    for (layer, name), row in weight_summary.iterrows():
        lines.append(
            f"| {layer} (`{name}`) | {row[('mean_weight', 'mean')]:.4f} ± "
            f"{row[('mean_weight', 'std')]:.4f} | "
            f"{row[('std_weight', 'mean')]:.4f} ± "
            f"{row[('std_weight', 'std')]:.4f} |"
        )
    lines.extend(
        [
            "",
            "Adjacent WavLM layers are highly correlated, so individual softmax weights",
            "are not uniquely identifiable. Interpret stable weight regions rather than",
            "a single layer's exact value.",
        ]
    )
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
