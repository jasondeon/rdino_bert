from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cached frozen RDINO embeddings with recording-level probes"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=40)
    return parser.parse_args()


def load_split(input_dir: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(input_dir / f"{split}_embeddings.npy", mmap_mode="r")
    metadata = pd.read_csv(
        input_dir / f"{split}_windows.csv", dtype={"subject_id": str}
    )
    if embeddings.ndim != 2 or len(embeddings) != len(metadata):
        raise ValueError(f"Invalid {split} embedding cache")
    expected = np.arange(len(metadata))
    if not np.array_equal(metadata["embedding_index"].to_numpy(), expected):
        raise ValueError(f"{split} metadata is not aligned with its embedding array")
    return embeddings, metadata


def aggregate_recordings(
    embeddings: np.ndarray, metadata: pd.DataFrame, include_std: bool
) -> tuple[np.ndarray, pd.DataFrame]:
    features = []
    rows = []
    for recording_index, group in metadata.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(dtype=int)
        values = np.asarray(embeddings[indices], dtype=np.float64)
        labels = group["regression_label"].to_numpy(dtype=float)
        if not np.allclose(labels, labels[0]):
            raise ValueError(f"Labels differ within recording {recording_index}")
        feature = values.mean(axis=0)
        if include_std:
            feature = np.concatenate([feature, values.std(axis=0, ddof=0)])
        features.append(feature)
        first = group.iloc[0]
        rows.append(
            {
                "recording_index": int(recording_index),
                "audio_path": first["audio_path"],
                "subject_id": str(first["subject_id"]),
                "regression_truth": float(labels[0]),
                "window_count": len(group),
            }
        )
    return np.stack(features), pd.DataFrame(rows)


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
        "prediction_target_sd_ratio": (
            prediction_std / truth_std if truth_std > 0 else float("nan")
        ),
        "mean_error": float(np.mean(prediction - truth)),
    }


def bootstrap_r2(
    truth: np.ndarray,
    prediction: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(truth), len(truth))
        if np.var(truth[indices]) > 0:
            values.append(r2_score(truth[indices], prediction[indices]))
    return tuple(np.percentile(values, [2.5, 97.5]))


def fit_probe(
    *,
    name: str,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    cv_folds: int,
    bootstrap_samples: int,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, float | str], np.ndarray]:
    train_truth = train_metadata["regression_truth"].to_numpy(dtype=float)
    validation_truth = validation_metadata["regression_truth"].to_numpy(dtype=float)
    groups = train_metadata["subject_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    folds = min(cv_folds, len(unique_groups))
    if folds < 2:
        raise ValueError("At least two training subjects are required for grouped CV")
    splitter = GroupKFold(n_splits=folds)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {"ridge__alpha": np.logspace(-6, 6, 25)},
        scoring="neg_root_mean_squared_error",
        cv=splitter,
        n_jobs=-1,
        refit=True,
    )
    search.fit(train_features, train_truth, groups=groups)
    prediction = search.predict(validation_features)
    result: dict[str, float | str] = {
        "probe": name,
        "feature_dimension": train_features.shape[1],
        "best_alpha": float(search.best_params_["ridge__alpha"]),
        "grouped_cv_rmse": float(-search.best_score_),
        **metrics(validation_truth, prediction),
    }
    low, high = bootstrap_r2(
        validation_truth,
        prediction,
        np.random.default_rng(seed),
        bootstrap_samples,
    )
    result["r2_ci_low"] = low
    result["r2_ci_high"] = high
    prediction_frame = validation_metadata.copy()
    prediction_frame["regression_prediction"] = prediction
    prediction_frame["regression_error"] = prediction - validation_truth
    prediction_frame.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    return result, prediction


def similarity_summary(
    embeddings: np.ndarray, metadata: pd.DataFrame, split: str
) -> dict[str, float | str]:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    centroid_rows = []
    within = []
    for _, group in metadata.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(dtype=int)
        centroid = values[indices].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        centroid_rows.append(centroid)
        within.extend((normalized[indices] @ centroid).tolist())
    centroids = np.stack(centroid_rows)
    similarities = centroids @ centroids.T
    between = similarities[np.triu_indices(len(centroids), k=1)]
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    explained = singular_values**2
    explained /= explained.sum()
    cumulative = np.cumsum(explained)
    return {
        "split": split,
        "windows": len(metadata),
        "recordings": len(centroids),
        "mean_windows_per_recording": float(
            metadata.groupby("recording_index").size().mean()
        ),
        "mean_window_to_recording_centroid_cosine": float(np.mean(within)),
        "mean_between_recording_centroid_cosine": float(np.mean(between)),
        "p95_between_recording_centroid_cosine": float(np.percentile(between, 95)),
        "principal_components_for_90_percent_variance": int(
            np.searchsorted(cumulative, 0.9) + 1
        ),
    }


def plot_predictions(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]
    limits = [float(min(truth.min(), *(p.min() for p in predictions.values()))),
              float(max(truth.max(), *(p.max() for p in predictions.values())))]
    for axis, (name, prediction) in zip(axes, predictions.items()):
        score = r2_score(truth, prediction)
        axis.scatter(truth, prediction, alpha=0.7)
        axis.plot(limits, limits, linestyle="--", color="black", linewidth=1)
        axis.set(title=f"{name}: R2={score:.3f}", xlabel="Observed", ylabel="Predicted")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    train_embeddings, train_windows = load_split(input_dir, "train")
    validation_embeddings, validation_windows = load_split(input_dir, "validation")
    overlap = set(train_windows["subject_id"].astype(str)) & set(
        validation_windows["subject_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"Subject leakage between train and validation caches: {sorted(overlap)[:10]}"
        )

    probe_rows = []
    prediction_arrays = {}
    validation_truth = None
    for name, include_std in (("mean", False), ("mean_std", True)):
        train_features, train_recordings = aggregate_recordings(
            train_embeddings, train_windows, include_std
        )
        validation_features, validation_recordings = aggregate_recordings(
            validation_embeddings, validation_windows, include_std
        )
        if validation_truth is None:
            validation_truth = validation_recordings["regression_truth"].to_numpy(
                dtype=float
            )
        row, prediction = fit_probe(
            name=name,
            train_features=train_features,
            validation_features=validation_features,
            train_metadata=train_recordings,
            validation_metadata=validation_recordings,
            cv_folds=args.cv_folds,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            output_dir=input_dir,
        )
        probe_rows.append(row)
        prediction_arrays[name] = prediction

    train_truth = train_recordings["regression_truth"].to_numpy(dtype=float)
    baseline_prediction = np.full_like(validation_truth, train_truth.mean())
    baseline = {
        "probe": "training_mean",
        "feature_dimension": 0,
        "best_alpha": float("nan"),
        "grouped_cv_rmse": float("nan"),
        **metrics(validation_truth, baseline_prediction),
    }
    low, high = bootstrap_r2(
        validation_truth,
        baseline_prediction,
        np.random.default_rng(args.seed),
        args.bootstrap_samples,
    )
    baseline["r2_ci_low"] = low
    baseline["r2_ci_high"] = high
    probe_rows.append(baseline)

    rng = np.random.default_rng(args.seed)
    differences = []
    for _ in range(args.bootstrap_samples):
        indices = rng.integers(0, len(validation_truth), len(validation_truth))
        if np.var(validation_truth[indices]) <= 0:
            continue
        differences.append(
            r2_score(validation_truth[indices], prediction_arrays["mean_std"][indices])
            - r2_score(validation_truth[indices], prediction_arrays["mean"][indices])
        )
    delta_low, delta_high = np.percentile(differences, [2.5, 97.5])
    delta = r2_score(validation_truth, prediction_arrays["mean_std"]) - r2_score(
        validation_truth, prediction_arrays["mean"]
    )

    probe_summary = pd.DataFrame(probe_rows)
    probe_summary.to_csv(input_dir / "probe_summary.csv", index=False)
    similarity = pd.DataFrame(
        [
            similarity_summary(train_embeddings, train_windows, "train"),
            similarity_summary(
                validation_embeddings, validation_windows, "validation"
            ),
        ]
    )
    similarity.to_csv(input_dir / "embedding_similarity.csv", index=False)
    plot_predictions(
        validation_truth,
        prediction_arrays,
        input_dir / "probe_predictions.png",
    )

    lines = [
        "# Frozen RDINO embedding audit",
        "",
        "Regularization was selected using subject-grouped cross-validation on the",
        "training split. The validation split was used only for final evaluation.",
        "",
        "| Probe | Features | Alpha | CV RMSE | Validation R2 | R2 95% CI | RMSE | ICC(2,1) | Pearson | Pred/target SD |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in probe_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['probe']} | {int(row['feature_dimension'])} | "
            f"{row['best_alpha']:.4g} | {row['grouped_cv_rmse']:.4f} | "
            f"{row['r2']:.4f} | [{row['r2_ci_low']:.4f}, {row['r2_ci_high']:.4f}] | "
            f"{row['rmse']:.4f} | {row['icc_2_1']:.4f} | {row['pearson']:.4f} | "
            f"{row['prediction_target_sd_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregation comparison",
            "",
            f"Mean+standard-deviation minus mean validation R2: {delta:.4f} "
            f"(paired bootstrap 95% CI [{delta_low:.4f}, {delta_high:.4f}]).",
            "",
            "## Embedding geometry",
            "",
            "| Split | Windows | Recordings | Windows/recording | Within-recording cosine | Between-recording cosine | Between p95 | PCs for 90% variance |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in similarity.to_dict(orient="records"):
        lines.append(
            f"| {row['split']} | {int(row['windows'])} | {int(row['recordings'])} | "
            f"{row['mean_windows_per_recording']:.2f} | "
            f"{row['mean_window_to_recording_centroid_cosine']:.4f} | "
            f"{row['mean_between_recording_centroid_cosine']:.4f} | "
            f"{row['p95_between_recording_centroid_cosine']:.4f} | "
            f"{int(row['principal_components_for_90_percent_variance'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- A useful ridge probe indicates that downstream optimization or aggregation",
            "  is the likely bottleneck rather than the frozen checkpoint.",
            "- If mean+standard-deviation clearly beats mean, preserve window-level",
            "  variability in the neural recording head.",
            "- High within-recording similarity with weak probes suggests stable speaker",
            "  identity embeddings that contain little target signal.",
        ]
    )
    report_path = input_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(probe_summary.to_string(index=False))
    print(similarity.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
