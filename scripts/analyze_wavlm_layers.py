from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_metrics import intraclass_correlation_2_1


POOLING_MODES = ("mean", "mean_temporal_std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit recording-level regression probes to every WavLM layer"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=-1)
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


def aggregate_recordings(
    means: np.ndarray, stds: np.ndarray, metadata: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    recording_means = []
    recording_temporal_stds = []
    rows = []
    for recording_index, group in metadata.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(dtype=int)
        labels = group["regression_label"].to_numpy(dtype=float)
        if not np.allclose(labels, labels[0]):
            raise ValueError(f"Labels differ within recording {recording_index}")
        recording_means.append(np.asarray(means[indices], dtype=np.float64).mean(0))
        recording_temporal_stds.append(
            np.asarray(stds[indices], dtype=np.float64).mean(0)
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
    return (
        np.stack(recording_means),
        np.stack(recording_temporal_stds),
        pd.DataFrame(rows),
    )


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


def bootstrap_r2(
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
    return tuple(float(value) for value in np.percentile(values, [2.5, 97.5]))


def features_for(
    means: np.ndarray, temporal_stds: np.ndarray, layer: int, pooling: str
) -> np.ndarray:
    if pooling == "mean":
        return means[:, layer, :]
    if pooling == "mean_temporal_std":
        return np.concatenate(
            [means[:, layer, :], temporal_stds[:, layer, :]], axis=1
        )
    raise ValueError(f"Unknown pooling mode: {pooling}")


def fit_probe(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    *,
    cv_folds: int,
    bootstrap_samples: int,
    seed: int,
    jobs: int,
) -> tuple[dict[str, float], np.ndarray]:
    train_truth = train_metadata["regression_truth"].to_numpy(dtype=float)
    validation_truth = validation_metadata["regression_truth"].to_numpy(dtype=float)
    groups = train_metadata["subject_id"].astype(str).to_numpy()
    folds = min(cv_folds, len(np.unique(groups)))
    if folds < 2:
        raise ValueError("At least two training subjects are required")
    search = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
        {"ridge__alpha": np.logspace(-6, 6, 25)},
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=folds),
        n_jobs=jobs,
        refit=True,
    )
    search.fit(train_features, train_truth, groups=groups)
    prediction = search.predict(validation_features)
    result = {
        "feature_dimension": int(train_features.shape[1]),
        "best_alpha": float(search.best_params_["ridge__alpha"]),
        "grouped_cv_rmse": float(-search.best_score_),
        **metrics(validation_truth, prediction),
    }
    low, high = bootstrap_r2(
        validation_truth, prediction, samples=bootstrap_samples, seed=seed
    )
    result["r2_ci_low"] = low
    result["r2_ci_high"] = high
    return result, prediction


def plot_layers(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    fields = (
        ("grouped_cv_rmse", "Grouped training CV", "RMSE"),
        ("r2", "Final validation", "R²"),
        ("rmse", "Final validation", "RMSE"),
        ("prediction_target_sd_ratio", "Prediction spread", "Pred/target SD"),
    )
    labels = {"mean": "Temporal mean", "mean_temporal_std": "Mean + temporal SD"}
    for axis, (field, title, ylabel) in zip(axes.flat, fields):
        for pooling in POOLING_MODES:
            rows = summary[summary["pooling"] == pooling].sort_values("layer_index")
            axis.plot(
                rows["layer_index"],
                rows[field],
                marker="o",
                label=labels[pooling],
            )
        if field == "r2":
            axis.axhline(0, color="black", linewidth=1, alpha=0.5)
        axis.set(title=title, xlabel="WavLM hidden-state index", ylabel=ylabel)
        axis.set_xticks(sorted(summary["layer_index"].unique()))
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2 or args.bootstrap_samples < 100:
        raise ValueError("Use at least 2 CV folds and 100 bootstrap samples")
    input_dir = args.input_dir.expanduser().resolve()
    config = json.loads((input_dir / "extraction_config.json").read_text())
    layer_names = list(config["layer_names"])
    train_means, train_stds, train_windows = load_split(input_dir, "train")
    validation_means, validation_stds, validation_windows = load_split(
        input_dir, "validation"
    )
    overlap = set(train_windows["subject_id"]) & set(validation_windows["subject_id"])
    if overlap:
        raise ValueError(f"Subject leakage in caches: {sorted(overlap)[:10]}")
    if train_means.shape[1] != len(layer_names):
        raise ValueError("Layer names do not match cached tensors")

    train_mean, train_temporal_std, train_recordings = aggregate_recordings(
        train_means, train_stds, train_windows
    )
    validation_mean, validation_temporal_std, validation_recordings = (
        aggregate_recordings(validation_means, validation_stds, validation_windows)
    )
    validation_truth = validation_recordings["regression_truth"].to_numpy(float)
    prediction_dir = input_dir / "layer_predictions"
    prediction_dir.mkdir(exist_ok=True)
    rows = []
    predictions: dict[tuple[int, str], np.ndarray] = {}
    for layer, layer_name in enumerate(layer_names):
        for pooling_index, pooling in enumerate(POOLING_MODES):
            result, prediction = fit_probe(
                features_for(train_mean, train_temporal_std, layer, pooling),
                features_for(
                    validation_mean, validation_temporal_std, layer, pooling
                ),
                train_recordings,
                validation_recordings,
                cv_folds=args.cv_folds,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + layer * len(POOLING_MODES) + pooling_index,
                jobs=args.jobs,
            )
            rows.append(
                {
                    "layer_index": layer,
                    "layer_name": layer_name,
                    "pooling": pooling,
                    **result,
                }
            )
            predictions[(layer, pooling)] = prediction
            prediction_frame = validation_recordings.copy()
            prediction_frame["regression_prediction"] = prediction
            prediction_frame["regression_error"] = prediction - validation_truth
            prediction_frame.to_csv(
                prediction_dir / f"layer_{layer:02d}_{pooling}.csv", index=False
            )
            print(
                f"layer={layer:02d} pooling={pooling} "
                f"cv_rmse={result['grouped_cv_rmse']:.4f} "
                f"validation_r2={result['r2']:.4f}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(input_dir / "layer_probe_summary.csv", index=False)
    selected = summary.loc[summary["grouped_cv_rmse"].idxmin()]
    selected_key = (int(selected["layer_index"]), str(selected["pooling"]))
    selected_predictions = validation_recordings.copy()
    selected_predictions["regression_prediction"] = predictions[selected_key]
    selected_predictions["regression_error"] = (
        selected_predictions["regression_prediction"] - validation_truth
    )
    selected_predictions.to_csv(input_dir / "selected_probe_predictions.csv", index=False)

    train_truth = train_recordings["regression_truth"].to_numpy(float)
    baseline_prediction = np.full_like(validation_truth, train_truth.mean())
    baseline = metrics(validation_truth, baseline_prediction)
    baseline_low, baseline_high = bootstrap_r2(
        validation_truth,
        baseline_prediction,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    plot_layers(summary, input_dir / "layer_probe_curves.png")

    lines = [
        "# Frozen WavLM Base+ layer audit",
        "",
        f"Model: `{config['model_name']}`. Hidden-state index 0 is the feature",
        "projection before the transformer blocks; indices 1–12 are transformer",
        "outputs. Ridge regularization and the selected layer/pooling configuration",
        "use subject-grouped cross-validation on training recordings only.",
        "",
        "## Training-CV-selected configuration",
        "",
        f"Layer {int(selected['layer_index'])} (`{selected['layer_name']}`), "
        f"pooling `{selected['pooling']}`: CV RMSE {selected['grouped_cv_rmse']:.4f}; "
        f"validation R² {selected['r2']:.4f} "
        f"[{selected['r2_ci_low']:.4f}, {selected['r2_ci_high']:.4f}], RMSE "
        f"{selected['rmse']:.4f}, ICC(2,1) {selected['icc_2_1']:.4f}, Pearson "
        f"{selected['pearson']:.4f}.",
        "",
        f"Training-mean baseline: validation R² {baseline['r2']:.4f} "
        f"[{baseline_low:.4f}, {baseline_high:.4f}], RMSE {baseline['rmse']:.4f}.",
        "",
        "## All layer probes",
        "",
        "| Layer | Name | Pooling | CV RMSE | Validation R² [95% CI] | RMSE | ICC | Pearson | Pred/target SD | Alpha |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.sort_values(["layer_index", "pooling"]).to_dict("records"):
        lines.append(
            f"| {row['layer_index']} | {row['layer_name']} | {row['pooling']} | "
            f"{row['grouped_cv_rmse']:.4f} | {row['r2']:.4f} "
            f"[{row['r2_ci_low']:.4f}, {row['r2_ci_high']:.4f}] | "
            f"{row['rmse']:.4f} | {row['icc_2_1']:.4f} | {row['pearson']:.4f} | "
            f"{row['prediction_target_sd_ratio']:.4f} | {row['best_alpha']:.4g} |"
        )
    lines.extend(
        [
            "",
            "Validation results for non-selected layers are exploratory. Do not choose",
            "a layer from the validation column without a new confirmation split or",
            "nested resampling.",
        ]
    )
    report_path = input_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
