from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_wavlm_domain_generalization import load_recordings


POOLING_MODES = ("mean", "mean_temporal_std")


@dataclass(frozen=True)
class Candidate:
    layer: int
    pooling: str
    alpha: float

    @property
    def name(self) -> str:
        return f"layer={self.layer},pooling={self.pooling},alpha={self.alpha:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether within-subject changes in cached WavLM embeddings "
            "predict week-8 minus baseline MADRS changes"
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--poolings", choices=POOLING_MODES, nargs="+", default=POOLING_MODES
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(100.0, 1000.0, 10000.0, 100000.0),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inner-site-folds", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--site-diagnostic-folds", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_visit(audio_path: str) -> str:
    match = re.search(r"[_-](00|08)[_-]", Path(audio_path).name)
    if match is None:
        raise ValueError(f"Could not parse visit from {audio_path}")
    return match.group(1)


def parse_site(subject_id: str) -> str:
    match = re.match(r"CBN17[_-]([^_-]+)", str(subject_id).strip().upper())
    if match is None:
        raise ValueError(f"Could not parse OPTIMUM-D site from {subject_id}")
    return match.group(1)


def make_pairs(
    means: np.ndarray, stds: np.ndarray, recordings: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    selected = recordings["subject_id"].str.startswith("CBN17", na=False)
    metadata = recordings[selected].copy()
    metadata["row_index"] = np.flatnonzero(selected.to_numpy())
    metadata["visit"] = metadata["audio_path"].map(parse_visit)
    metadata["site"] = metadata["subject_id"].map(parse_site)
    baseline = metadata[metadata["visit"] == "00"].set_index("subject_id")
    week8 = metadata[metadata["visit"] == "08"].set_index("subject_id")
    subjects = sorted(set(baseline.index) & set(week8.index))
    rows = []
    baseline_indices = []
    week8_indices = []
    for subject in subjects:
        first = baseline.loc[subject]
        second = week8.loc[subject]
        if isinstance(first, pd.DataFrame) or isinstance(second, pd.DataFrame):
            raise ValueError(f"Duplicate visit for {subject}")
        if first["site"] != second["site"]:
            raise ValueError(f"Site changes between visits for {subject}")
        baseline_indices.append(int(first["row_index"]))
        week8_indices.append(int(second["row_index"]))
        baseline_madrs = float(first["regression_truth"])
        week8_madrs = float(second["regression_truth"])
        rows.append(
            {
                "subject_id": subject,
                "site": str(first["site"]),
                "baseline_audio_path": str(first["audio_path"]),
                "week8_audio_path": str(second["audio_path"]),
                "baseline_windows": int(first["window_count"]),
                "week8_windows": int(second["window_count"]),
                "baseline_madrs": baseline_madrs,
                "week8_madrs": week8_madrs,
                "delta_madrs": week8_madrs - baseline_madrs,
            }
        )
    first = np.asarray(baseline_indices, dtype=int)
    second = np.asarray(week8_indices, dtype=int)
    return means[first], means[second], stds[first], stds[second], pd.DataFrame(rows)


def candidate_features(
    baseline_means: np.ndarray,
    week8_means: np.ndarray,
    baseline_stds: np.ndarray,
    week8_stds: np.ndarray,
    layers: list[int],
    poolings: list[str],
) -> dict[tuple[int, str], np.ndarray]:
    result = {}
    for layer in layers:
        mean_delta = week8_means[:, layer] - baseline_means[:, layer]
        for pooling in poolings:
            if pooling == "mean":
                values = mean_delta
            else:
                std_delta = week8_stds[:, layer] - baseline_stds[:, layer]
                values = np.concatenate([mean_delta, std_delta], axis=1)
            result[(layer, pooling)] = np.asarray(values, dtype=np.float64)
    return result


def ridge_predict(
    train_features: np.ndarray,
    train_truth: np.ndarray,
    test_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)
    model = Ridge(alpha=alpha)
    model.fit(scaled_train, train_truth)
    return model.predict(scaled_test)


def baseline_predict(
    train_baseline: np.ndarray,
    train_truth: np.ndarray,
    test_baseline: np.ndarray,
) -> np.ndarray:
    model = LinearRegression()
    model.fit(train_baseline[:, None], train_truth)
    return model.predict(test_baseline[:, None])


def combined_predict(
    train_features: np.ndarray,
    train_baseline: np.ndarray,
    train_truth: np.ndarray,
    test_features: np.ndarray,
    test_baseline: np.ndarray,
    alpha: float,
) -> np.ndarray:
    baseline_model = LinearRegression()
    baseline_model.fit(train_baseline[:, None], train_truth)
    residual = train_truth - baseline_model.predict(train_baseline[:, None])
    return baseline_model.predict(test_baseline[:, None]) + ridge_predict(
        train_features, residual, test_features, alpha
    )


def inner_splits(indices: np.ndarray, sites: np.ndarray, fold_count: int):
    inner_sites = sites[indices]
    count = min(fold_count, len(np.unique(inner_sites)))
    splitter = GroupKFold(n_splits=count)
    for fit_relative, held_relative in splitter.split(indices, groups=inner_sites):
        yield indices[fit_relative], indices[held_relative]


def select_candidate(
    feature_sets: dict[tuple[int, str], np.ndarray],
    truth: np.ndarray,
    baseline: np.ndarray,
    sites: np.ndarray,
    train_indices: np.ndarray,
    candidates: list[Candidate],
    mode: str,
    inner_site_folds: int,
) -> Candidate:
    best_candidate = None
    best_rmse = float("inf")
    splits = list(inner_splits(train_indices, sites, inner_site_folds))
    for candidate in candidates:
        features = feature_sets[(candidate.layer, candidate.pooling)]
        prediction = np.full(len(truth), np.nan)
        for fit, held_out in splits:
            if mode == "acoustic":
                prediction[held_out] = ridge_predict(
                    features[fit], truth[fit], features[held_out], candidate.alpha
                )
            elif mode == "combined":
                prediction[held_out] = combined_predict(
                    features[fit],
                    baseline[fit],
                    truth[fit],
                    features[held_out],
                    baseline[held_out],
                    candidate.alpha,
                )
            else:
                raise ValueError(mode)
        inner_held_out = np.concatenate([held_out for _, held_out in splits])
        rmse = float(
            np.sqrt(mean_squared_error(truth[inner_held_out], prediction[inner_held_out]))
        )
        if rmse < best_rmse:
            best_rmse = rmse
            best_candidate = candidate
    if best_candidate is None:
        raise RuntimeError("No candidate was selected")
    return best_candidate


def evaluate_outer_folds(
    feature_sets: dict[tuple[int, str], np.ndarray],
    pairs: pd.DataFrame,
    candidates: list[Candidate],
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    evaluation: str,
    inner_site_folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pairs["delta_madrs"].to_numpy(float)
    baseline = pairs["baseline_madrs"].to_numpy(float)
    sites = pairs["site"].to_numpy(str)
    prediction_sums = {
        "mean_change": np.zeros(len(truth)),
        "baseline_madrs": np.zeros(len(truth)),
        "delta_embedding": np.zeros(len(truth)),
        "baseline_plus_delta_embedding": np.zeros(len(truth)),
    }
    counts = np.zeros(len(truth), dtype=int)
    acoustic_selections = []
    combined_selections = []
    for fold_index, (fit, held_out) in enumerate(folds, start=1):
        counts[held_out] += 1
        prediction_sums["mean_change"][held_out] += truth[fit].mean()
        prediction_sums["baseline_madrs"][held_out] += baseline_predict(
            baseline[fit], truth[fit], baseline[held_out]
        )
        acoustic = select_candidate(
            feature_sets,
            truth,
            baseline,
            sites,
            fit,
            candidates,
            "acoustic",
            inner_site_folds,
        )
        combined = select_candidate(
            feature_sets,
            truth,
            baseline,
            sites,
            fit,
            candidates,
            "combined",
            inner_site_folds,
        )
        acoustic_selections.append(acoustic.name)
        combined_selections.append(combined.name)
        acoustic_features = feature_sets[(acoustic.layer, acoustic.pooling)]
        combined_features = feature_sets[(combined.layer, combined.pooling)]
        prediction_sums["delta_embedding"][held_out] += ridge_predict(
            acoustic_features[fit],
            truth[fit],
            acoustic_features[held_out],
            acoustic.alpha,
        )
        prediction_sums["baseline_plus_delta_embedding"][held_out] += combined_predict(
            combined_features[fit],
            baseline[fit],
            truth[fit],
            combined_features[held_out],
            baseline[held_out],
            combined.alpha,
        )
        print(
            f"evaluation={evaluation} fold={fold_index}/{len(folds)} "
            f"acoustic=({acoustic.name}) combined=({combined.name})",
            flush=True,
        )
    if np.any(counts == 0):
        raise RuntimeError(f"Missing outer predictions for {evaluation}")
    prediction_frame = pairs[["subject_id", "site", "delta_madrs"]].copy()
    rows = []
    for model_name, total in prediction_sums.items():
        prediction = total / counts
        prediction_frame[model_name] = prediction
        row = {"evaluation": evaluation, "model": model_name, **metrics(truth, prediction)}
        if model_name == "delta_embedding":
            row["selection_counts"] = json.dumps(Counter(acoustic_selections).most_common())
        elif model_name == "baseline_plus_delta_embedding":
            row["selection_counts"] = json.dumps(Counter(combined_selections).most_common())
        else:
            row["selection_counts"] = "[]"
        rows.append(row)
    return pd.DataFrame(rows), prediction_frame


def correlation(first: np.ndarray, second: np.ndarray, rank: bool = False) -> float:
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan")
    function = spearmanr if rank else pearsonr
    return float(function(first, second).statistic)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "pearson": correlation(truth, prediction),
        "spearman": correlation(truth, prediction, rank=True),
        "prediction_target_sd_ratio": float(
            np.std(prediction, ddof=1) / np.std(truth, ddof=1)
        ),
        "mean_error": float(np.mean(prediction - truth)),
    }


def paired_bootstrap(
    truth: np.ndarray,
    baseline_prediction: np.ndarray,
    model_prediction: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    model_metrics = {"r2": [], "rmse": [], "pearson": []}
    differences = {"r2_difference": [], "rmse_difference": []}
    for _ in range(samples):
        selected = rng.integers(0, len(truth), len(truth))
        if np.var(truth[selected]) < 1e-12:
            continue
        baseline = metrics(truth[selected], baseline_prediction[selected])
        model = metrics(truth[selected], model_prediction[selected])
        for key in model_metrics:
            model_metrics[key].append(model[key])
        differences["r2_difference"].append(model["r2"] - baseline["r2"])
        differences["rmse_difference"].append(model["rmse"] - baseline["rmse"])
    result = {}
    for key, values in {**model_metrics, **differences}.items():
        low, high = np.nanpercentile(values, [2.5, 97.5])
        result[f"{key}_ci_low"] = float(low)
        result[f"{key}_ci_high"] = float(high)
    return result


def cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.sum(first * second, axis=1) / np.maximum(denominator, 1e-12)


def retrieval_accuracy(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second_norm = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    nearest = np.argmax(first_norm @ second_norm.T, axis=1)
    return float(np.mean(nearest == np.arange(len(first))))


def site_predictability(
    features: np.ndarray,
    sites: np.ndarray,
    folds: int,
    seed: int,
    max_iterations: int,
) -> float:
    values = sorted(np.unique(sites))
    labels = np.asarray([values.index(value) for value in sites], dtype=int)
    prediction = np.full(len(labels), -1)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold_index, (fit, held_out) in enumerate(splitter.split(features, labels), start=1):
        scaler = StandardScaler()
        scaled_fit = scaler.fit_transform(features[fit])
        scaled_held_out = scaler.transform(features[held_out])
        model = LogisticRegression(
            C=0.001,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=max_iterations,
            random_state=seed + fold_index,
        )
        model.fit(scaled_fit, labels[fit])
        prediction[held_out] = model.predict(scaled_held_out)
    return float(balanced_accuracy_score(labels, prediction))


def representation_diagnostics(
    baseline_means: np.ndarray,
    week8_means: np.ndarray,
    feature_sets: dict[tuple[int, str], np.ndarray],
    pairs: pd.DataFrame,
    layers: list[int],
    folds: int,
    seed: int,
    max_iterations: int,
) -> pd.DataFrame:
    delta_madrs = pairs["delta_madrs"].to_numpy(float)
    absolute_change = np.abs(delta_madrs)
    sites = pairs["site"].to_numpy(str)
    rows = []
    for layer in layers:
        first = np.asarray(baseline_means[:, layer], dtype=np.float64)
        second = np.asarray(week8_means[:, layer], dtype=np.float64)
        similarity = cosine_rows(first, second)
        delta_norm = np.linalg.norm(second - first, axis=1)
        for pooling in POOLING_MODES:
            if (layer, pooling) not in feature_sets:
                continue
            rows.append(
                {
                    "layer_index": layer,
                    "pooling": pooling,
                    "same_subject_cosine_mean": float(np.mean(similarity)),
                    "same_subject_cosine_sd": float(np.std(similarity, ddof=1)),
                    "cross_visit_retrieval_accuracy": retrieval_accuracy(first, second),
                    "delta_norm_signed_madrs_spearman": correlation(
                        delta_norm, delta_madrs, rank=True
                    ),
                    "delta_norm_absolute_madrs_spearman": correlation(
                        delta_norm, absolute_change, rank=True
                    ),
                    "delta_embedding_site_balanced_accuracy": site_predictability(
                        feature_sets[(layer, pooling)],
                        sites,
                        folds,
                        seed + layer,
                        max_iterations,
                    ),
                }
            )
    return pd.DataFrame(rows)


def add_intervals(
    summary: pd.DataFrame,
    prediction_frames: dict[str, pd.DataFrame],
    pairs: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    truth = pairs["delta_madrs"].to_numpy(float)
    result = summary.copy()
    for row_index, row in result.iterrows():
        frame = prediction_frames[str(row["evaluation"])]
        baseline = frame["baseline_madrs"].to_numpy(float)
        prediction = frame[str(row["model"])].to_numpy(float)
        baseline_metrics = metrics(truth, baseline)
        current_metrics = metrics(truth, prediction)
        result.loc[row_index, "r2_difference"] = (
            current_metrics["r2"] - baseline_metrics["r2"]
        )
        result.loc[row_index, "rmse_difference"] = (
            current_metrics["rmse"] - baseline_metrics["rmse"]
        )
        intervals = paired_bootstrap(
            truth, baseline, prediction, samples, seed + row_index
        )
        for key, value in intervals.items():
            result.loc[row_index, key] = value
    return result


def plot_results(
    summary: pd.DataFrame,
    prediction_frames: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
    pairs: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    model_order = [
        "mean_change",
        "baseline_madrs",
        "delta_embedding",
        "baseline_plus_delta_embedding",
    ]
    for axis, evaluation in zip(axes[0], ("repeated_subject_cv", "leave_one_site_out")):
        rows = summary[summary["evaluation"] == evaluation].set_index("model").loc[model_order]
        axis.bar(np.arange(len(rows)), rows["r2"])
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(np.arange(len(rows)), ["Mean", "Baseline", "Δembed", "Baseline+Δ"], rotation=20)
        axis.set(title=evaluation.replace("_", " ").title(), ylabel="R²")
        axis.grid(axis="y", alpha=0.25)
    site_frame = prediction_frames["leave_one_site_out"]
    truth = pairs["delta_madrs"].to_numpy(float)
    axes[1, 0].scatter(truth, site_frame["baseline_plus_delta_embedding"], alpha=0.7)
    limits = [min(truth.min(), site_frame["baseline_plus_delta_embedding"].min()), max(truth.max(), site_frame["baseline_plus_delta_embedding"].max())]
    axes[1, 0].plot(limits, limits, color="black", linestyle="--")
    axes[1, 0].set(title="Leave-one-site-out combined model", xlabel="Observed ΔMADRS", ylabel="Predicted ΔMADRS")
    for pooling in POOLING_MODES:
        rows = diagnostics[diagnostics["pooling"] == pooling].sort_values("layer_index")
        axes[1, 1].plot(rows["layer_index"], rows["delta_embedding_site_balanced_accuracy"], marker="o", label=pooling)
    axes[1, 1].axhline(1 / pairs["site"].nunique(), color="black", linestyle="--", label="Chance")
    axes[1, 1].set(title="Site information remaining after differencing", xlabel="Layer", ylabel="Balanced accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extraction = json.loads((input_dir / "extraction_config.json").read_text())
    layer_names = list(extraction["layer_names"])
    if any(layer < 0 or layer >= len(layer_names) for layer in args.layers):
        raise ValueError(f"Layers must be between 0 and {len(layer_names) - 1}")
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": extraction["model_name"],
        "window_seconds": extraction["window_seconds"],
        "stride_seconds": extraction["stride_seconds"],
        "layers": args.layers,
        "layer_names": {str(layer): layer_names[layer] for layer in args.layers},
        "poolings": args.poolings,
        "ridge_alphas": args.ridge_alphas,
        "outer_subject_folds": args.folds,
        "outer_subject_repeats": args.repeats,
        "inner_selection": f"{args.inner_site_folds}-fold site-grouped CV",
        "primary_evaluation": "leave-one-site-out",
        "change_direction": "week8 minus baseline",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; cached arrays were not loaded.")
        return

    means, stds, recordings = load_recordings(input_dir, "train")
    baseline_means, week8_means, baseline_stds, week8_stds, pairs = make_pairs(
        means, stds, recordings
    )
    feature_sets = candidate_features(
        baseline_means,
        week8_means,
        baseline_stds,
        week8_stds,
        args.layers,
        args.poolings,
    )
    candidates = [
        Candidate(layer, pooling, alpha)
        for layer in args.layers
        for pooling in args.poolings
        for alpha in args.ridge_alphas
    ]
    repeated_folds = []
    for repeat in range(args.repeats):
        splitter = KFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed + repeat,
        )
        repeated_folds.extend(splitter.split(pairs))
    sites = pairs["site"].to_numpy(str)
    site_folds = list(
        GroupKFold(n_splits=len(np.unique(sites))).split(pairs, groups=sites)
    )
    repeated_summary, repeated_predictions = evaluate_outer_folds(
        feature_sets,
        pairs,
        candidates,
        repeated_folds,
        evaluation="repeated_subject_cv",
        inner_site_folds=args.inner_site_folds,
    )
    site_summary, site_predictions = evaluate_outer_folds(
        feature_sets,
        pairs,
        candidates,
        site_folds,
        evaluation="leave_one_site_out",
        inner_site_folds=args.inner_site_folds,
    )
    prediction_frames = {
        "repeated_subject_cv": repeated_predictions,
        "leave_one_site_out": site_predictions,
    }
    summary = add_intervals(
        pd.concat([repeated_summary, site_summary], ignore_index=True),
        prediction_frames,
        pairs,
        args.bootstrap_samples,
        args.seed,
    )
    diagnostics = representation_diagnostics(
        baseline_means,
        week8_means,
        feature_sets,
        pairs,
        args.layers,
        args.site_diagnostic_folds,
        args.seed,
        args.max_iterations,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    pairs.to_csv(output_dir / "paired_subjects.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    repeated_predictions.to_csv(output_dir / "repeated_cv_predictions.csv", index=False)
    site_predictions.to_csv(output_dir / "site_loso_predictions.csv", index=False)
    diagnostics.to_csv(output_dir / "representation_diagnostics.csv", index=False)
    plot_results(
        summary,
        prediction_frames,
        diagnostics,
        pairs,
        output_dir / "longitudinal_wavlm.png",
    )

    lines = [
        f"# Longitudinal WavLM audit: {extraction['model_name']}",
        "",
        f"Paired OPTIMUM-D subjects: **{len(pairs)}** across {pairs['site'].nunique()} sites.",
        f"Cached embeddings use {extraction['window_seconds']:g}-second windows and a "
        f"{extraction['stride_seconds']:g}-second stride.",
        "All changes are week 8 minus baseline. Layer, pooling, and Ridge alpha",
        "are selected inside each outer fold using site-grouped inner CV.",
        "",
        "## Nested cross-validation",
        "",
        "| Evaluation | Model | R² [95% CI] | RMSE | Pearson [95% CI] | ΔR² vs baseline [95% CI] | ΔRMSE vs baseline [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["evaluation", "rmse"]).to_dict("records"):
        lines.append(
            f"| {row['evaluation']} | {row['model']} | {row['r2']:.3f} "
            f"[{row['r2_ci_low']:.3f}, {row['r2_ci_high']:.3f}] | "
            f"{row['rmse']:.3f} | {row['pearson']:.3f} "
            f"[{row['pearson_ci_low']:.3f}, {row['pearson_ci_high']:.3f}] | "
            f"{row['r2_difference']:.3f} [{row['r2_difference_ci_low']:.3f}, "
            f"{row['r2_difference_ci_high']:.3f}] | {row['rmse_difference']:.3f} "
            f"[{row['rmse_difference_ci_low']:.3f}, {row['rmse_difference_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Representation diagnostics",
            "",
            "| Layer | Pooling | Same-subject cosine | Cross-visit retrieval | Δnorm vs |ΔMADRS| Spearman | Site balanced accuracy |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics.to_dict("records"):
        lines.append(
            f"| {int(row['layer_index'])} | {row['pooling']} | "
            f"{row['same_subject_cosine_mean']:.3f} | "
            f"{row['cross_visit_retrieval_accuracy']:.3f} | "
            f"{row['delta_norm_absolute_madrs_spearman']:.3f} | "
            f"{row['delta_embedding_site_balanced_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`selection_counts` in `model_summary.csv` shows how often each",
            "representation was selected. Consistent selection across outer folds is",
            "evidence of a stable layer; diffuse selection is evidence of weak signal.",
            "The leave-one-site-out combined model is the primary result.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
