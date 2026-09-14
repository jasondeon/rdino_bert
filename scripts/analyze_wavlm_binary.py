from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold


POOLING_MODES = ("mean", "mean_temporal_std")
AGGREGATIONS = ("mean", "top_quartile")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit binary MADRS signal in cached WavLM layers using recording-"
            "pooled and recording-balanced window-level logistic probes"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/wavlm-base-plus-layer-audit"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wavlm-base-plus-binary-audit"),
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=(20, 23))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_split(
    input_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    means = np.load(input_dir / f"{split}_layer_means.npy", mmap_mode="r")
    stds = np.load(input_dir / f"{split}_layer_stds.npy", mmap_mode="r")
    metadata = pd.read_csv(
        input_dir / f"{split}_windows.csv", dtype={"subject_id": str}
    )
    if means.ndim != 3 or stds.shape != means.shape or len(means) != len(metadata):
        raise ValueError(f"Invalid or misaligned {split} WavLM cache")
    if not np.array_equal(
        metadata["embedding_index"].to_numpy(), np.arange(len(metadata))
    ):
        raise ValueError(f"Nonsequential {split} embedding indices")
    return means, stds, metadata


def recording_metadata(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for recording_index, group in windows.groupby("recording_index", sort=True):
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
                "window_count": len(group),
            }
        )
    return pd.DataFrame(rows)


def features_for(
    means: np.ndarray, stds: np.ndarray, layer: int, pooling: str
) -> np.ndarray:
    mean_features = np.asarray(means[:, layer, :], dtype=np.float32)
    if pooling == "mean":
        return mean_features
    if pooling == "mean_temporal_std":
        return np.concatenate(
            [mean_features, np.asarray(stds[:, layer, :], dtype=np.float32)], axis=1
        )
    raise ValueError(f"Unknown pooling: {pooling}")


def aggregate_feature_rows(
    window_features: np.ndarray,
    windows: pd.DataFrame,
    recordings: pd.DataFrame,
) -> np.ndarray:
    by_recording = windows.groupby("recording_index", sort=False).indices
    return np.stack(
        [
            window_features[np.asarray(by_recording[index], dtype=int)].mean(axis=0)
            for index in recordings["recording_index"]
        ]
    )


def class_balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both binary classes must be present in every training fold")
    weights = len(labels) / (2.0 * counts)
    return weights[labels]


def weighted_standardize(
    train: np.ndarray, test: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.average(train, axis=0, weights=weights)
    variance = np.average(np.square(train - mean), axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def fit_probabilities(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    c_value: float,
    sample_weights: np.ndarray,
    seed: int,
    max_iterations: int,
) -> np.ndarray:
    scaled_train, scaled_test = weighted_standardize(
        train_features, test_features, sample_weights
    )
    # The dual formulation is much faster for recording-level probes, where the
    # feature dimension exceeds the number of examples. Window probes use primal.
    dual = len(scaled_train) < scaled_train.shape[1]
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        dual=dual,
        max_iter=max_iterations,
        random_state=seed,
    )
    model.fit(scaled_train, train_labels, sample_weight=sample_weights)
    return model.predict_proba(scaled_test)[:, 1]


def make_folds(
    labels: np.ndarray, groups: np.ndarray, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    count = min(folds, len(np.unique(groups)))
    if count < 2:
        raise ValueError("At least two subjects are required")
    splitter = StratifiedGroupKFold(
        n_splits=count, shuffle=True, random_state=seed
    )
    result = list(splitter.split(np.zeros(len(labels)), labels, groups))
    for train_indices, held_out_indices in result:
        if len(np.unique(labels[train_indices])) != 2:
            raise ValueError("A grouped training fold contains only one class")
        if set(groups[train_indices]) & set(groups[held_out_indices]):
            raise RuntimeError("Subject leakage within cross-validation")
    return result


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return (
        float(roc_auc_score(labels, probabilities))
        if len(np.unique(labels)) == 2
        else float("nan")
    )


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": safe_auc(labels, probabilities),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def choose_cutoff(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    scores = true_positive[finite] - false_positive[finite]
    candidates = thresholds[finite][np.isclose(scores, scores.max())]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, cutoff: float
) -> dict[str, float]:
    predictions = (probabilities >= cutoff).astype(int)
    negative = labels == 0
    positive = labels == 1
    return {
        **probability_metrics(labels, probabilities),
        "cutoff": float(cutoff),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "sensitivity": float(recall_score(labels, predictions, pos_label=1)),
        "specificity": float(np.mean(predictions[negative] == 0)),
        "precision": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "predicted_positive_rate": float(np.mean(predictions)),
        "positive_rate": float(np.mean(positive)),
    }


def select_recording_probe(
    train_features: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_values: list[float],
    *,
    seed: int,
    max_iterations: int,
) -> tuple[float, np.ndarray, dict[str, float]]:
    best: tuple[float, float, float, np.ndarray] | None = None
    for c_value in c_values:
        oof = np.full(len(labels), np.nan)
        for fold_index, (fit_indices, held_out_indices) in enumerate(folds, start=1):
            fit_labels = labels[fit_indices]
            oof[held_out_indices] = fit_probabilities(
                train_features[fit_indices],
                fit_labels,
                train_features[held_out_indices],
                c_value=c_value,
                sample_weights=class_balanced_weights(fit_labels),
                seed=seed + fold_index,
                max_iterations=max_iterations,
            )
        current = probability_metrics(labels, oof)
        key = (current["roc_auc"], current["average_precision"])
        if best is None or key > best[:2]:
            best = (*key, c_value, oof)
    if best is None:
        raise RuntimeError("No recording probe was fit")
    _, _, best_c, best_oof = best
    return best_c, best_oof, probability_metrics(labels, best_oof)


def recording_window_indices(
    windows: pd.DataFrame, recording_indices: set[int]
) -> np.ndarray:
    return np.flatnonzero(windows["recording_index"].isin(recording_indices).to_numpy())


def window_weights(
    windows: pd.DataFrame,
    recording_labels: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    recording_ids = windows["recording_index"].to_numpy(int)
    labels = np.asarray([recording_labels[value] for value in recording_ids], dtype=int)
    counts = windows["recording_index"].value_counts().to_dict()
    unique_labels = np.asarray(list(recording_labels.values()), dtype=int)
    recording_class_weights = class_balanced_weights(unique_labels)
    class_weight = {
        int(label): float(recording_class_weights[np.flatnonzero(unique_labels == label)[0]])
        for label in (0, 1)
    }
    weights = np.asarray(
        [class_weight[label] / counts[recording] for recording, label in zip(recording_ids, labels)],
        dtype=float,
    )
    weights *= len(weights) / weights.sum()
    return labels, weights


def aggregate_probabilities(
    window_probabilities: np.ndarray,
    windows: pd.DataFrame,
    recordings: pd.DataFrame,
    mode: str,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "recording_index": windows["recording_index"].to_numpy(int),
            "probability": window_probabilities,
        }
    )
    grouped = frame.groupby("recording_index", sort=False)["probability"]
    if mode == "mean":
        values = grouped.mean().to_dict()
    elif mode == "top_quartile":
        values = grouped.apply(
            lambda value: float(
                np.sort(value.to_numpy())[-max(1, math.ceil(len(value) / 4)) :].mean()
            )
        ).to_dict()
    else:
        raise ValueError(f"Unknown aggregation: {mode}")
    return np.asarray([values[index] for index in recordings["recording_index"]])


def select_window_probe(
    window_features: np.ndarray,
    windows: pd.DataFrame,
    recordings: pd.DataFrame,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_values: list[float],
    *,
    seed: int,
    max_iterations: int,
) -> tuple[float, str, np.ndarray, dict[str, float]]:
    recording_labels = dict(zip(recordings["recording_index"], labels))
    best: tuple[float, float, float, str, np.ndarray] | None = None
    for c_value in c_values:
        oof_by_aggregation = {
            mode: np.full(len(recordings), np.nan) for mode in AGGREGATIONS
        }
        for fold_index, (fit_recordings, held_out_recordings) in enumerate(
            folds, start=1
        ):
            fit_ids = set(recordings.iloc[fit_recordings]["recording_index"].astype(int))
            held_out_ids = set(
                recordings.iloc[held_out_recordings]["recording_index"].astype(int)
            )
            fit_windows = recording_window_indices(windows, fit_ids)
            held_out_windows = recording_window_indices(windows, held_out_ids)
            fold_windows = windows.iloc[fit_windows]
            fold_labels = {index: recording_labels[index] for index in fit_ids}
            fit_labels, fit_weights = window_weights(fold_windows, fold_labels)
            probabilities = fit_probabilities(
                window_features[fit_windows],
                fit_labels,
                window_features[held_out_windows],
                c_value=c_value,
                sample_weights=fit_weights,
                seed=seed + fold_index,
                max_iterations=max_iterations,
            )
            held_out_window_frame = windows.iloc[held_out_windows]
            held_out_recording_frame = recordings.iloc[held_out_recordings]
            for mode in AGGREGATIONS:
                oof_by_aggregation[mode][held_out_recordings] = aggregate_probabilities(
                    probabilities,
                    held_out_window_frame,
                    held_out_recording_frame,
                    mode,
                )
        for mode, oof in oof_by_aggregation.items():
            current = probability_metrics(labels, oof)
            key = (current["roc_auc"], current["average_precision"])
            if best is None or key > best[:2]:
                best = (*key, c_value, mode, oof)
    if best is None:
        raise RuntimeError("No window probe was fit")
    _, _, best_c, best_mode, best_oof = best
    return best_c, best_mode, best_oof, probability_metrics(labels, best_oof)


def clustered_bootstrap(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    grouped = {
        subject: indices.to_numpy(int)
        for subject, indices in frame.groupby("subject_id").groups.items()
    }
    subjects = np.asarray(list(grouped), dtype=object)
    values: dict[str, list[float]] = {
        "roc_auc": [],
        "average_precision": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "mcc": [],
    }
    for _ in range(samples):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([grouped[subject] for subject in sampled])
        labels = frame["binary_truth"].to_numpy(int)[indices]
        if len(np.unique(labels)) < 2:
            continue
        probabilities = frame["probability"].to_numpy(float)[indices]
        cutoff = float(frame["cutoff"].iloc[0])
        current = classification_metrics(labels, probabilities, cutoff)
        for key in values:
            values[key].append(current[key])
    return {
        key: [float(value) for value in np.percentile(metric_values, [2.5, 97.5])]
        for key, metric_values in values.items()
    }


def prediction_frame(
    recordings: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    cutoff: float,
) -> pd.DataFrame:
    result = recordings.copy()
    result["binary_truth"] = labels
    result["probability"] = probabilities
    result["cutoff"] = cutoff
    result["binary_prediction"] = (probabilities >= cutoff).astype(int)
    return result


def plot_layer_results(summary: pd.DataFrame, output_path: Path) -> None:
    thresholds = sorted(summary["madrs_threshold"].unique())
    figure, axes = plt.subplots(
        len(thresholds), 2, figsize=(13, 5 * len(thresholds)), squeeze=False,
        constrained_layout=True,
    )
    labels = {"mean": "Temporal mean", "mean_temporal_std": "Mean + temporal SD"}
    for row_index, threshold in enumerate(thresholds):
        subset = summary[summary["madrs_threshold"] == threshold]
        for column_index, (field, title) in enumerate(
            (("cv_roc_auc", "Training grouped CV"), ("validation_roc_auc", "External validation"))
        ):
            axis = axes[row_index, column_index]
            for pooling in POOLING_MODES:
                rows = subset[subset["pooling"] == pooling].sort_values("layer_index")
                axis.plot(rows["layer_index"], rows[field], marker="o", label=labels[pooling])
            axis.axhline(0.5, color="black", linestyle="--", alpha=0.5)
            axis.set(
                title=f"MADRS ≥{threshold:g}: {title}",
                xlabel="WavLM hidden-state index",
                ylabel="AUROC",
                xticks=sorted(subset["layer_index"].unique()),
            )
            axis.grid(alpha=0.25)
            axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2 or args.bootstrap_samples < 100:
        raise ValueError("Use at least 2 folds and 100 bootstrap samples")
    if not args.thresholds or not args.c_values or any(value <= 0 for value in args.c_values):
        raise ValueError("Thresholds and positive C values are required")
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extraction = json.loads((input_dir / "extraction_config.json").read_text())
    layer_names = list(extraction["layer_names"])
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": extraction["model_name"],
        "thresholds": args.thresholds,
        "cv_folds": args.cv_folds,
        "c_values": args.c_values,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "selection_metric": "training subject-grouped OOF AUROC",
        "external_validation_used_for_selection": False,
        "window_loss_weighting": "equal total weight per recording and class",
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; cached arrays were not loaded.")
        return

    train_means, train_stds, train_windows = load_split(input_dir, "train")
    validation_means, validation_stds, validation_windows = load_split(
        input_dir, "validation"
    )
    if train_means.shape[1] != len(layer_names):
        raise ValueError("Layer names do not match cached features")
    if set(train_windows["subject_id"]) & set(validation_windows["subject_id"]):
        raise ValueError("Training and validation caches share subjects")
    train_recordings = recording_metadata(train_windows)
    validation_recordings = recording_metadata(validation_windows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    layer_rows = []
    selected_rows = []
    all_metrics: dict[str, object] = {"configuration": config, "thresholds": {}}
    for threshold_index, threshold in enumerate(args.thresholds):
        threshold_name = f"madrs_ge_{threshold:g}".replace(".", "p")
        threshold_dir = output_dir / threshold_name
        threshold_dir.mkdir(exist_ok=True)
        train_labels = (
            train_recordings["regression_truth"].to_numpy(float) >= threshold
        ).astype(int)
        validation_labels = (
            validation_recordings["regression_truth"].to_numpy(float) >= threshold
        ).astype(int)
        groups = train_recordings["subject_id"].astype(str).to_numpy()
        folds = make_folds(
            train_labels, groups, args.cv_folds, args.seed + threshold_index
        )

        configurations: dict[tuple[int, str], dict[str, object]] = {}
        for layer, layer_name in enumerate(layer_names):
            for pooling in POOLING_MODES:
                train_window_features = features_for(train_means, train_stds, layer, pooling)
                validation_window_features = features_for(
                    validation_means, validation_stds, layer, pooling
                )
                train_features = aggregate_feature_rows(
                    train_window_features, train_windows, train_recordings
                )
                validation_features = aggregate_feature_rows(
                    validation_window_features,
                    validation_windows,
                    validation_recordings,
                )
                best_c, oof, cv_metrics = select_recording_probe(
                    train_features,
                    train_labels,
                    folds,
                    list(args.c_values),
                    seed=args.seed + layer * 10,
                    max_iterations=args.max_iterations,
                )
                validation_probability = fit_probabilities(
                    train_features,
                    train_labels,
                    validation_features,
                    c_value=best_c,
                    sample_weights=class_balanced_weights(train_labels),
                    seed=args.seed,
                    max_iterations=args.max_iterations,
                )
                validation_probability_metrics = probability_metrics(
                    validation_labels, validation_probability
                )
                row = {
                    "madrs_threshold": threshold,
                    "layer_index": layer,
                    "layer_name": layer_name,
                    "pooling": pooling,
                    "feature_dimension": train_features.shape[1],
                    "best_c": best_c,
                    **{f"cv_{key}": value for key, value in cv_metrics.items()},
                    **{
                        f"validation_{key}": value
                        for key, value in validation_probability_metrics.items()
                    },
                }
                layer_rows.append(row)
                configurations[(layer, pooling)] = {
                    "row": row,
                    "oof": oof,
                    "validation_probability": validation_probability,
                }
                print(
                    f"threshold={threshold:g} layer={layer:02d} pooling={pooling} "
                    f"cv_auc={cv_metrics['roc_auc']:.4f} "
                    f"validation_auc={validation_probability_metrics['roc_auc']:.4f}",
                    flush=True,
                )

        selected_key = max(
            configurations,
            key=lambda key: (
                configurations[key]["row"]["cv_roc_auc"],
                configurations[key]["row"]["cv_average_precision"],
            ),
        )
        selected = configurations[selected_key]
        selected_row = selected["row"]
        selected_train_window_features = features_for(
            train_means, train_stds, selected_key[0], selected_key[1]
        )
        selected_validation_window_features = features_for(
            validation_means, validation_stds, selected_key[0], selected_key[1]
        )
        recording_cutoff = choose_cutoff(train_labels, selected["oof"])
        recording_validation = prediction_frame(
            validation_recordings,
            validation_labels,
            selected["validation_probability"],
            recording_cutoff,
        )
        recording_oof = prediction_frame(
            train_recordings, train_labels, selected["oof"], recording_cutoff
        )
        recording_validation.to_csv(
            threshold_dir / "recording_probe_validation_predictions.csv", index=False
        )
        recording_oof.to_csv(
            threshold_dir / "recording_probe_oof_predictions.csv", index=False
        )
        recording_metrics = classification_metrics(
            validation_labels, selected["validation_probability"], recording_cutoff
        )
        recording_ci = clustered_bootstrap(
            recording_validation,
            samples=args.bootstrap_samples,
            seed=args.seed + threshold_index * 100,
        )

        window_c, aggregation, window_oof_probability, window_cv = select_window_probe(
            selected_train_window_features,
            train_windows,
            train_recordings,
            train_labels,
            folds,
            list(args.c_values),
            seed=args.seed + 1000,
            max_iterations=args.max_iterations,
        )
        full_recording_labels = dict(
            zip(train_recordings["recording_index"], train_labels)
        )
        window_labels, full_window_weights = window_weights(
            train_windows, full_recording_labels
        )
        validation_window_probability = fit_probabilities(
            selected_train_window_features,
            window_labels,
            selected_validation_window_features,
            c_value=window_c,
            sample_weights=full_window_weights,
            seed=args.seed,
            max_iterations=args.max_iterations,
        )
        window_validation_probability = aggregate_probabilities(
            validation_window_probability,
            validation_windows,
            validation_recordings,
            aggregation,
        )
        window_cutoff = choose_cutoff(train_labels, window_oof_probability)
        window_oof = prediction_frame(
            train_recordings, train_labels, window_oof_probability, window_cutoff
        )
        window_validation = prediction_frame(
            validation_recordings,
            validation_labels,
            window_validation_probability,
            window_cutoff,
        )
        window_oof.to_csv(
            threshold_dir / "window_probe_oof_predictions.csv", index=False
        )
        window_validation.to_csv(
            threshold_dir / "window_probe_validation_predictions.csv", index=False
        )
        window_metrics = classification_metrics(
            validation_labels, window_validation_probability, window_cutoff
        )
        window_ci = clustered_bootstrap(
            window_validation,
            samples=args.bootstrap_samples,
            seed=args.seed + threshold_index * 100 + 1,
        )

        selected_rows.extend(
            [
                {
                    "madrs_threshold": threshold,
                    "model": "recording_pooled",
                    "layer_index": selected_key[0],
                    "layer_name": layer_names[selected_key[0]],
                    "pooling": selected_key[1],
                    "c_value": selected_row["best_c"],
                    "aggregation": "recording_mean_features",
                    "cv_roc_auc": selected_row["cv_roc_auc"],
                    **{f"validation_{key}": value for key, value in recording_metrics.items()},
                },
                {
                    "madrs_threshold": threshold,
                    "model": "window_level",
                    "layer_index": selected_key[0],
                    "layer_name": layer_names[selected_key[0]],
                    "pooling": selected_key[1],
                    "c_value": window_c,
                    "aggregation": aggregation,
                    "cv_roc_auc": window_cv["roc_auc"],
                    **{f"validation_{key}": value for key, value in window_metrics.items()},
                },
            ]
        )
        all_metrics["thresholds"][threshold_name] = {
            "train_class_counts": np.bincount(train_labels, minlength=2).tolist(),
            "validation_class_counts": np.bincount(
                validation_labels, minlength=2
            ).tolist(),
            "selected_layer": int(selected_key[0]),
            "selected_layer_name": layer_names[selected_key[0]],
            "selected_pooling": selected_key[1],
            "recording_pooled": {
                "c_value": selected_row["best_c"],
                "training_oof": classification_metrics(
                    train_labels, selected["oof"], recording_cutoff
                ),
                "validation": recording_metrics,
                "validation_subject_bootstrap_ci": recording_ci,
            },
            "window_level": {
                "c_value": window_c,
                "aggregation": aggregation,
                "training_oof": classification_metrics(
                    train_labels, window_oof_probability, window_cutoff
                ),
                "validation": window_metrics,
                "validation_subject_bootstrap_ci": window_ci,
            },
        }
        print(
            f"SELECTED threshold={threshold:g}: layer={selected_key[0]} "
            f"pooling={selected_key[1]}; recording validation AUC="
            f"{recording_metrics['roc_auc']:.4f}; window validation AUC="
            f"{window_metrics['roc_auc']:.4f}",
            flush=True,
        )

    layer_summary = pd.DataFrame(layer_rows)
    selected_summary = pd.DataFrame(selected_rows)
    layer_summary.to_csv(output_dir / "layer_probe_summary.csv", index=False)
    selected_summary.to_csv(output_dir / "selected_model_summary.csv", index=False)
    plot_layer_results(layer_summary, output_dir / "layer_probe_auc.png")
    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")

    lines = [
        "# WavLM binary MADRS signal audit",
        "",
        f"Frozen cache: `{extraction['model_name']}`. Model and aggregation selection",
        "used subject-grouped training cross-validation only. External validation was",
        "not used for selection. Window losses give equal total weight to every",
        "recording and then balance the two classes. Confidence intervals resample",
        "validation subjects as clusters.",
        "",
        "Training CV values are selection scores, not unbiased performance estimates,",
        "because the layer and pooling mode were selected from those scores.",
        "",
        "| MADRS | Model | Selected representation | Aggregation | CV AUROC | Validation AUROC | 95% CI | AP | Balanced accuracy | Macro F1 | MCC |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        threshold_name = f"madrs_ge_{row['madrs_threshold']:g}".replace(".", "p")
        model_key = "recording_pooled" if row["model"] == "recording_pooled" else "window_level"
        ci = all_metrics["thresholds"][threshold_name][model_key][
            "validation_subject_bootstrap_ci"
        ]["roc_auc"]
        lines.append(
            f"| ≥{row['madrs_threshold']:g} | {row['model']} | layer "
            f"{int(row['layer_index'])} {row['pooling']} | {row['aggregation']} | "
            f"{row['cv_roc_auc']:.3f} | {row['validation_roc_auc']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {row['validation_average_precision']:.3f} | "
            f"{row['validation_balanced_accuracy']:.3f} | "
            f"{row['validation_macro_f1']:.3f} | {row['validation_mcc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "AUROC and average precision are threshold-free. The classification cutoff",
            "was chosen from training OOF predictions by Youden's index and then held",
            "fixed for external validation.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
