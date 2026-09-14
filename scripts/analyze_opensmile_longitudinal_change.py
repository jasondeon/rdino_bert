from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relate within-subject changes in cached OpenSMILE IS09 features to "
            "week-8 minus baseline MADRS changes for paired CBN17 recordings"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/opensmile-is09-manuscript-cohort"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/opensmile-is09-longitudinal-change"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    )
    parser.add_argument(
        "--elasticnet-alphas",
        type=float,
        nargs="+",
        default=(0.001, 0.01, 0.1, 1.0, 10.0),
    )
    parser.add_argument(
        "--elasticnet-l1-ratios",
        type=float,
        nargs="+",
        default=(0.1, 0.5, 0.9),
    )
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--rf-min-leaf", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_visit(audio_path: str) -> str:
    match = re.search(r"[_-](00|08)[_-]", Path(audio_path).name)
    if match is None:
        raise ValueError(f"Could not parse CBN17 visit from {audio_path}")
    return match.group(1)


def parse_site(subject_id: str) -> str:
    match = re.match(r"CBN17[_-]([^_-]+)", str(subject_id).strip().upper())
    if match is None:
        raise ValueError(f"Could not parse CBN17 site from {subject_id}")
    return match.group(1)


def load_recording_features(
    input_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    features = np.load(input_dir / "train_features.npy", mmap_mode="r")
    windows = pd.read_csv(
        input_dir / "train_windows.csv", dtype={"subject_id": str}
    )
    feature_names = pd.read_csv(input_dir / "feature_names.csv")[
        "feature_name"
    ].astype(str).tolist()
    if features.ndim != 2 or features.shape[1] != len(feature_names):
        raise ValueError("Feature cache and feature names are inconsistent")
    if len(features) != len(windows):
        raise ValueError("Feature cache and window metadata are misaligned")
    if not np.array_equal(
        windows["embedding_index"].to_numpy(), np.arange(len(windows))
    ):
        raise ValueError("Embedding indices are not sequential")

    rows = []
    recording_features = []
    for recording_index, group in windows.groupby("recording_index", sort=True):
        first = group.iloc[0]
        subject_id = str(first["subject_id"]).strip()
        if not subject_id.upper().startswith("CBN17"):
            continue
        labels = group["regression_label"].to_numpy(float)
        if not np.allclose(labels, labels[0]):
            raise ValueError(f"Labels differ within recording {recording_index}")
        indices = group["embedding_index"].to_numpy(int)
        recording_features.append(
            np.asarray(features[indices], dtype=np.float64).mean(axis=0)
        )
        rows.append(
            {
                "recording_index": int(recording_index),
                "subject_id": subject_id,
                "site": parse_site(subject_id),
                "visit": parse_visit(str(first["audio_path"])),
                "audio_path": str(first["audio_path"]),
                "madrs": float(labels[0]),
                "window_count": len(group),
            }
        )
    return np.stack(recording_features), pd.DataFrame(rows), feature_names


def make_pairs(
    recording_features: np.ndarray, recordings: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    feature_by_recording = {
        int(recording): recording_features[index]
        for index, recording in enumerate(recordings["recording_index"])
    }
    baseline_rows = recordings[recordings["visit"] == "00"].set_index("subject_id")
    week8_rows = recordings[recordings["visit"] == "08"].set_index("subject_id")
    subjects = sorted(set(baseline_rows.index) & set(week8_rows.index))
    rows = []
    baseline = []
    week8 = []
    for subject in subjects:
        first = baseline_rows.loc[subject]
        second = week8_rows.loc[subject]
        if isinstance(first, pd.DataFrame) or isinstance(second, pd.DataFrame):
            raise ValueError(f"Duplicate visit for {subject}")
        if first["site"] != second["site"]:
            raise ValueError(f"Site changed between visits for {subject}")
        baseline.append(feature_by_recording[int(first["recording_index"])])
        week8.append(feature_by_recording[int(second["recording_index"])])
        baseline_madrs = float(first["madrs"])
        week8_madrs = float(second["madrs"])
        rows.append(
            {
                "subject_id": subject,
                "site": str(first["site"]),
                "baseline_recording_index": int(first["recording_index"]),
                "week8_recording_index": int(second["recording_index"]),
                "baseline_audio_path": str(first["audio_path"]),
                "week8_audio_path": str(second["audio_path"]),
                "baseline_windows": int(first["window_count"]),
                "week8_windows": int(second["window_count"]),
                "baseline_madrs": baseline_madrs,
                "week8_madrs": week8_madrs,
                "delta_madrs": week8_madrs - baseline_madrs,
                "response_50_percent": bool(
                    baseline_madrs > 0
                    and week8_madrs <= 0.5 * baseline_madrs
                ),
                "remission_madrs_le_10": bool(week8_madrs <= 10),
            }
        )
    return np.stack(baseline), np.stack(week8), pd.DataFrame(rows)


def finite_correlation(function, first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return float("nan"), 1.0
    value, p_value = function(first, second)
    return float(value), float(p_value)


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    result = np.full(len(p_values), np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if not len(finite):
        return result
    order = finite[np.argsort(p_values[finite])]
    adjusted = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def residualize(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(covariate)), covariate])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return values - design @ coefficients


def feature_statistics(
    baseline: np.ndarray,
    week8: np.ndarray,
    delta_madrs: np.ndarray,
    baseline_madrs: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    change = week8 - baseline
    target_residual = residualize(delta_madrs, baseline_madrs)
    rows = []
    for index, name in enumerate(feature_names):
        pearson, pearson_p = finite_correlation(
            pearsonr, change[:, index], delta_madrs
        )
        spearman, spearman_p = finite_correlation(
            spearmanr, change[:, index], delta_madrs
        )
        stability, stability_p = finite_correlation(
            pearsonr, baseline[:, index], week8[:, index]
        )
        feature_residual = residualize(change[:, index], baseline_madrs)
        partial, partial_p = finite_correlation(
            pearsonr, feature_residual, target_residual
        )
        rows.append(
            {
                "feature_index": index,
                "feature_name": name,
                "delta_mean": float(np.mean(change[:, index])),
                "delta_sd": float(np.std(change[:, index], ddof=1)),
                "baseline_week8_pearson": stability,
                "baseline_week8_p": stability_p,
                "delta_pearson": pearson,
                "delta_pearson_p": pearson_p,
                "delta_spearman": spearman,
                "delta_spearman_p": spearman_p,
                "baseline_adjusted_delta_pearson": partial,
                "baseline_adjusted_delta_pearson_p": partial_p,
            }
        )
    frame = pd.DataFrame(rows)
    for p_column in (
        "delta_pearson_p",
        "delta_spearman_p",
        "baseline_adjusted_delta_pearson_p",
    ):
        frame[p_column.removesuffix("_p") + "_fdr"] = fdr_bh(
            frame[p_column].to_numpy(float)
        )
    return frame


def metric_values(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth_sd = float(np.std(truth, ddof=1))
    prediction_sd = float(np.std(prediction, ddof=1))
    pearson, _ = finite_correlation(pearsonr, truth, prediction)
    spearman, _ = finite_correlation(spearmanr, truth, prediction)
    return {
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "pearson": pearson,
        "spearman": spearman,
        "prediction_target_sd_ratio": prediction_sd / truth_sd,
        "mean_error": float(np.mean(prediction - truth)),
    }


def inner_cv(
    sample_count: int,
    folds: int,
    seed: int,
    groups: np.ndarray | None,
):
    if groups is not None:
        return GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    return KFold(n_splits=min(folds, sample_count), shuffle=True, random_state=seed)


def fit_tuned_model(
    kind: str,
    train_features: np.ndarray,
    train_truth: np.ndarray,
    test_features: np.ndarray,
    *,
    groups: np.ndarray | None,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, str]:
    cv = inner_cv(len(train_truth), args.inner_folds, seed, groups)
    if kind == "ridge":
        estimator = Pipeline([("scale", StandardScaler()), ("model", Ridge())])
        parameters = {"model__alpha": args.ridge_alphas}
    elif kind == "elasticnet":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(max_iter=20_000, random_state=seed),
                ),
            ]
        )
        parameters = {
            "model__alpha": args.elasticnet_alphas,
            "model__l1_ratio": args.elasticnet_l1_ratios,
        }
    else:
        raise ValueError(kind)
    search = GridSearchCV(
        estimator,
        parameters,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=args.jobs,
        refit=True,
    )
    fit_kwargs = {"groups": groups} if groups is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search.fit(train_features, train_truth, **fit_kwargs)
    return search.predict(test_features), json.dumps(search.best_params_, sort_keys=True)


def evaluate_folds(
    delta_features: np.ndarray,
    pairs: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    name: str,
    inner_grouped: bool,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pairs["delta_madrs"].to_numpy(float)
    baseline_madrs = pairs["baseline_madrs"].to_numpy(float)[:, None]
    combined = np.concatenate([baseline_madrs, delta_features], axis=1)
    feature_sets = {"acoustic": delta_features, "baseline_plus_acoustic": combined}
    predictions = {
        model: np.zeros(len(truth), dtype=float)
        for model in (
            "mean_change",
            "baseline_madrs",
            "ridge_acoustic",
            "ridge_baseline_plus_acoustic",
            "elasticnet_acoustic",
            "elasticnet_baseline_plus_acoustic",
            "random_forest_acoustic",
            "random_forest_baseline_plus_acoustic",
        )
    }
    prediction_counts = np.zeros(len(truth), dtype=int)
    selections: dict[str, list[str]] = {key: [] for key in predictions}
    sites = pairs["site"].to_numpy(str)
    for fold_index, (fit, held_out) in enumerate(folds, start=1):
        prediction_counts[held_out] += 1
        predictions["mean_change"][held_out] += truth[fit].mean()
        simple = LinearRegression().fit(baseline_madrs[fit], truth[fit])
        predictions["baseline_madrs"][held_out] += simple.predict(
            baseline_madrs[held_out]
        )
        inner_groups = sites[fit] if inner_grouped else None
        for kind in ("ridge", "elasticnet"):
            for feature_name, values in feature_sets.items():
                key = f"{kind}_{feature_name}"
                prediction, selection = fit_tuned_model(
                    kind,
                    values[fit],
                    truth[fit],
                    values[held_out],
                    groups=inner_groups,
                    args=args,
                    seed=args.seed + fold_index,
                )
                predictions[key][held_out] += prediction
                selections[key].append(selection)
        for feature_name, values in feature_sets.items():
            key = f"random_forest_{feature_name}"
            model = RandomForestRegressor(
                n_estimators=args.rf_trees,
                max_features="sqrt",
                min_samples_leaf=args.rf_min_leaf,
                random_state=args.seed + fold_index,
                n_jobs=args.jobs,
            )
            model.fit(values[fit], truth[fit])
            predictions[key][held_out] += model.predict(values[held_out])
    if np.any(prediction_counts == 0):
        raise RuntimeError(f"Some subjects have no {name} prediction")

    rows = []
    prediction_frame = pairs[["subject_id", "site", "delta_madrs"]].copy()
    for model_name, prediction_sum in predictions.items():
        prediction = prediction_sum / prediction_counts
        prediction_frame[model_name] = prediction
        rows.append(
            {
                "evaluation": name,
                "model": model_name,
                **metric_values(truth, prediction),
                "selected_parameters": json.dumps(
                    Counter(selections[model_name]).most_common(), sort_keys=True
                ),
            }
        )
    return pd.DataFrame(rows), prediction_frame


def bootstrap_intervals(
    truth: np.ndarray,
    prediction: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    r2_values = []
    correlation_values = []
    for _ in range(samples):
        selected = rng.integers(0, len(truth), len(truth))
        if np.var(truth[selected]) < 1e-12:
            continue
        current = metric_values(truth[selected], prediction[selected])
        r2_values.append(current["r2"])
        correlation_values.append(current["pearson"])
    r2_low, r2_high = np.nanpercentile(r2_values, [2.5, 97.5])
    pearson_low, pearson_high = np.nanpercentile(correlation_values, [2.5, 97.5])
    return {
        "r2_ci_low": float(r2_low),
        "r2_ci_high": float(r2_high),
        "pearson_ci_low": float(pearson_low),
        "pearson_ci_high": float(pearson_high),
    }


def plot_results(
    pairs: pd.DataFrame,
    feature_summary: pd.DataFrame,
    random_predictions: pd.DataFrame,
    site_predictions: pd.DataFrame,
    random_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    best_model = str(random_summary.sort_values("rmse").iloc[0]["model"])
    truth = pairs["delta_madrs"].to_numpy(float)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].hist(truth, bins=20, edgecolor="black")
    axes[0, 0].axvline(0, color="black", linestyle="--")
    axes[0, 0].set(title="MADRS change", xlabel="Week 8 − baseline", ylabel="Subjects")
    for axis, frame, title in (
        (axes[0, 1], random_predictions, "Repeated subject CV"),
        (axes[1, 0], site_predictions, "Leave-one-site-out"),
    ):
        axis.scatter(truth, frame[best_model], alpha=0.7)
        limits = [min(truth.min(), frame[best_model].min()), max(truth.max(), frame[best_model].max())]
        axis.plot(limits, limits, color="black", linestyle="--")
        axis.set(title=f"{title}: {best_model}", xlabel="Observed ΔMADRS", ylabel="Predicted ΔMADRS")
    top = feature_summary.reindex(
        feature_summary["baseline_adjusted_delta_pearson"].abs().sort_values().tail(12).index
    )
    colors = np.where(top["baseline_adjusted_delta_pearson"] >= 0, "tab:red", "tab:blue")
    axes[1, 1].barh(top["feature_name"], top["baseline_adjusted_delta_pearson"], color=colors)
    axes[1, 1].set(title="Largest baseline-adjusted feature-change correlations", xlabel="Partial Pearson r")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extraction = json.loads((input_dir / "extraction_config.json").read_text())
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "feature_set": extraction.get("feature_set"),
        "window_seconds": extraction.get("window_seconds"),
        "stride_seconds": extraction.get("stride_seconds"),
        "recording_aggregation": "mean of all cached window-level IS09 functionals",
        "change_direction": "week8 minus baseline",
        "folds": args.folds,
        "repeats": args.repeats,
        "inner_folds": args.inner_folds,
        "ridge_alphas": list(args.ridge_alphas),
        "elasticnet_alphas": list(args.elasticnet_alphas),
        "elasticnet_l1_ratios": list(args.elasticnet_l1_ratios),
        "rf_trees": args.rf_trees,
        "rf_min_leaf": args.rf_min_leaf,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; cached arrays were not loaded.")
        return

    recording_features, recordings, feature_names = load_recording_features(input_dir)
    baseline, week8, pairs = make_pairs(recording_features, recordings)
    if len(pairs) < args.folds:
        raise ValueError("Too few paired subjects")
    delta_features = week8 - baseline
    delta_madrs = pairs["delta_madrs"].to_numpy(float)
    feature_summary = feature_statistics(
        baseline,
        week8,
        delta_madrs,
        pairs["baseline_madrs"].to_numpy(float),
        feature_names,
    )

    repeated_folds = []
    for repeat in range(args.repeats):
        splitter = KFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed + repeat,
        )
        repeated_folds.extend(splitter.split(delta_features))
    random_summary, random_predictions = evaluate_folds(
        delta_features,
        pairs,
        repeated_folds,
        name="repeated_subject_cv",
        inner_grouped=False,
        args=args,
    )
    sites = pairs["site"].to_numpy(str)
    site_folds = list(GroupKFold(n_splits=len(np.unique(sites))).split(delta_features, delta_madrs, sites))
    site_summary, site_predictions = evaluate_folds(
        delta_features,
        pairs,
        site_folds,
        name="leave_one_site_out",
        inner_grouped=True,
        args=args,
    )
    summary = pd.concat([random_summary, site_summary], ignore_index=True)
    for row_index, row in summary.iterrows():
        frame = random_predictions if row["evaluation"] == "repeated_subject_cv" else site_predictions
        summary.loc[row_index, list(bootstrap_intervals(
            delta_madrs,
            frame[str(row["model"])].to_numpy(float),
            args.bootstrap_samples,
            args.seed + row_index,
        ).keys())] = list(bootstrap_intervals(
            delta_madrs,
            frame[str(row["model"])].to_numpy(float),
            args.bootstrap_samples,
            args.seed + row_index,
        ).values())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    pairs.to_csv(output_dir / "paired_subjects.csv", index=False)
    feature_summary.to_csv(output_dir / "feature_change_statistics.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    random_predictions.to_csv(output_dir / "repeated_cv_predictions.csv", index=False)
    site_predictions.to_csv(output_dir / "site_loso_predictions.csv", index=False)
    plot_results(
        pairs,
        feature_summary,
        random_predictions,
        site_predictions,
        random_summary,
        output_dir / "longitudinal_change.png",
    )

    baseline_delta_r, baseline_delta_p = finite_correlation(
        pearsonr, pairs["baseline_madrs"].to_numpy(float), delta_madrs
    )
    significant = feature_summary[
        feature_summary["baseline_adjusted_delta_pearson_fdr"] < 0.05
    ]
    top_features = feature_summary.reindex(
        feature_summary["baseline_adjusted_delta_pearson"].abs().sort_values(ascending=False).index
    ).head(15)
    lines = [
        "# OpenSMILE IS09 within-subject MADRS change audit",
        "",
        f"Paired CBN17 subjects: **{len(pairs)}** across {pairs['site'].nunique()} sites.",
        f"Features are recording-level means of {extraction.get('window_seconds'):g}-second",
        f"windows with a {extraction.get('stride_seconds'):g}-second stride.",
        "All changes are week 8 minus baseline, so negative ΔMADRS denotes improvement.",
        "",
        "## Cohort change",
        "",
        f"- Baseline MADRS: mean {pairs['baseline_madrs'].mean():.2f}, SD {pairs['baseline_madrs'].std():.2f}.",
        f"- Week-8 MADRS: mean {pairs['week8_madrs'].mean():.2f}, SD {pairs['week8_madrs'].std():.2f}.",
        f"- ΔMADRS: mean {delta_madrs.mean():.2f}, SD {delta_madrs.std(ddof=1):.2f}, range {delta_madrs.min():.0f} to {delta_madrs.max():.0f}.",
        f"- Baseline MADRS versus ΔMADRS: r={baseline_delta_r:.3f}, p={baseline_delta_p:.3g}.",
        f"- 50% response: {pairs['response_50_percent'].sum()}/{len(pairs)}; week-8 MADRS ≤10: {pairs['remission_madrs_le_10'].sum()}/{len(pairs)}.",
        "",
        "## Cross-validated ΔMADRS prediction",
        "",
        "| Evaluation | Model | R² [95% CI] | RMSE | MAE | Pearson [95% CI] | Pred/target SD |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["evaluation", "rmse"]).to_dict("records"):
        lines.append(
            f"| {row['evaluation']} | {row['model']} | {row['r2']:.3f} "
            f"[{row['r2_ci_low']:.3f}, {row['r2_ci_high']:.3f}] | {row['rmse']:.3f} | "
            f"{row['mae']:.3f} | {row['pearson']:.3f} "
            f"[{row['pearson_ci_low']:.3f}, {row['pearson_ci_high']:.3f}] | "
            f"{row['prediction_target_sd_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Feature-change associations",
            "",
            f"Features significant at FDR <0.05 after adjustment for baseline MADRS: **{len(significant)}**/{len(feature_summary)}.",
            "",
            "| Feature | Partial r | Raw p | FDR q | Baseline/week-8 stability |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_features.to_dict("records"):
        lines.append(
            f"| {row['feature_name']} | {row['baseline_adjusted_delta_pearson']:.3f} | "
            f"{row['baseline_adjusted_delta_pearson_p']:.3g} | "
            f"{row['baseline_adjusted_delta_pearson_fdr']:.3g} | "
            f"{row['baseline_week8_pearson']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Random subject CV estimates interpolation within OPTIMUM-D. Leave-one-site-out",
            "is the more relevant robustness check. Model comparisons are diagnostic, and",
            "the same paired cohort must not be treated as an untouched confirmation set.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
