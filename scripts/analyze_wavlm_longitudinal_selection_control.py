from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_wavlm_domain_generalization import load_recordings
from analyze_wavlm_longitudinal_change import (
    Candidate,
    candidate_features,
    inner_splits,
    make_pairs,
    metrics,
    paired_bootstrap,
    ridge_predict,
)
from analyze_wavlm_longitudinal_pairing_control import within_site_derangement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Selection-aware longitudinal WavLM permutation test with a "
            "recording-window-count control"
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=int, nargs="+")
    parser.add_argument(
        "--poolings",
        choices=("mean", "mean_temporal_std"),
        nargs="+",
        default=("mean", "mean_temporal_std"),
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(100.0, 1000.0, 10000.0, 100000.0),
    )
    parser.add_argument("--inner-site-folds", type=int, default=4)
    parser.add_argument("--shuffles", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def recording_covariates(pairs: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            pairs["baseline_madrs"].to_numpy(float),
            np.log1p(pairs["baseline_windows"].to_numpy(float)),
            np.log1p(pairs["week8_windows"].to_numpy(float)),
        ]
    )


def linear_predict(
    train_features: np.ndarray,
    train_truth: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    model = LinearRegression()
    model.fit(train_features, train_truth)
    return model.predict(test_features)


def adjusted_embedding_predict(
    train_embedding: np.ndarray,
    train_covariates: np.ndarray,
    train_truth: np.ndarray,
    test_embedding: np.ndarray,
    test_covariates: np.ndarray,
    alpha: float,
) -> np.ndarray:
    nuisance = LinearRegression()
    nuisance.fit(train_covariates, train_truth)
    residual = train_truth - nuisance.predict(train_covariates)
    return nuisance.predict(test_covariates) + ridge_predict(
        train_embedding, residual, test_embedding, alpha
    )


def select_candidate(
    feature_sets: dict[tuple[int, str], np.ndarray],
    truth: np.ndarray,
    covariates: np.ndarray,
    sites: np.ndarray,
    train_indices: np.ndarray,
    candidates: list[Candidate],
    inner_site_folds: int,
) -> Candidate:
    splits = list(inner_splits(train_indices, sites, inner_site_folds))
    held_out_indices = np.concatenate([held_out for _, held_out in splits])
    best_candidate = None
    best_rmse = float("inf")
    for candidate in candidates:
        features = feature_sets[(candidate.layer, candidate.pooling)]
        prediction = np.full(len(truth), np.nan)
        for fit, held_out in splits:
            prediction[held_out] = adjusted_embedding_predict(
                features[fit],
                covariates[fit],
                truth[fit],
                features[held_out],
                covariates[held_out],
                candidate.alpha,
            )
        rmse = metrics(
            truth[held_out_indices], prediction[held_out_indices]
        )["rmse"]
        if rmse < best_rmse:
            best_rmse = rmse
            best_candidate = candidate
    if best_candidate is None:
        raise RuntimeError("No WavLM candidate was selected")
    return best_candidate


def evaluate_embedding_loso(
    feature_sets: dict[tuple[int, str], np.ndarray],
    pairs: pd.DataFrame,
    candidates: list[Candidate],
    inner_site_folds: int,
    *,
    verbose: bool,
    covariates: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    truth = pairs["delta_madrs"].to_numpy(float)
    if covariates is None:
        covariates = recording_covariates(pairs)
    sites = pairs["site"].to_numpy(str)
    prediction = np.full(len(pairs), np.nan)
    selections = []
    splitter = GroupKFold(n_splits=len(np.unique(sites)))
    folds = list(splitter.split(covariates, groups=sites))
    for fold_index, (fit, held_out) in enumerate(folds, start=1):
        candidate = select_candidate(
            feature_sets,
            truth,
            covariates,
            sites,
            fit,
            candidates,
            inner_site_folds,
        )
        features = feature_sets[(candidate.layer, candidate.pooling)]
        prediction[held_out] = adjusted_embedding_predict(
            features[fit],
            covariates[fit],
            truth[fit],
            features[held_out],
            covariates[held_out],
            candidate.alpha,
        )
        selections.append(candidate.name)
        if verbose:
            held_site = str(np.unique(sites[held_out])[0])
            print(
                f"observed held_site={held_site} selected=({candidate.name})",
                flush=True,
            )
    if np.isnan(prediction).any():
        raise RuntimeError("Missing leave-one-site-out embedding predictions")
    return prediction, selections


def evaluate_linear_loso(
    features: np.ndarray, pairs: pd.DataFrame
) -> np.ndarray:
    truth = pairs["delta_madrs"].to_numpy(float)
    sites = pairs["site"].to_numpy(str)
    prediction = np.full(len(pairs), np.nan)
    splitter = GroupKFold(n_splits=len(np.unique(sites)))
    for fit, held_out in splitter.split(features, groups=sites):
        prediction[held_out] = linear_predict(
            features[fit], truth[fit], features[held_out]
        )
    return prediction


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def per_site_metrics(
    pairs: pd.DataFrame,
    baseline_prediction: np.ndarray,
    metadata_prediction: np.ndarray,
    embedding_prediction: np.ndarray,
) -> pd.DataFrame:
    frame = pairs[["site", "delta_madrs"]].copy()
    frame["baseline_madrs"] = baseline_prediction
    frame["baseline_plus_window_metadata"] = metadata_prediction
    frame["baseline_plus_window_metadata_plus_embedding"] = embedding_prediction
    rows = []
    for site, group in frame.groupby("site"):
        truth = group["delta_madrs"].to_numpy(float)
        baseline = metrics(truth, group["baseline_madrs"].to_numpy(float))
        metadata = metrics(
            truth, group["baseline_plus_window_metadata"].to_numpy(float)
        )
        embedding = metrics(
            truth,
            group["baseline_plus_window_metadata_plus_embedding"].to_numpy(float),
        )
        rows.append(
            {
                "site": site,
                "subjects": len(group),
                "baseline_r2": baseline["r2"],
                "metadata_r2": metadata["r2"],
                "embedding_r2": embedding["r2"],
                "embedding_delta_r2_vs_metadata": embedding["r2"]
                - metadata["r2"],
                "baseline_rmse": baseline["rmse"],
                "metadata_rmse": metadata["rmse"],
                "embedding_rmse": embedding["rmse"],
                "embedding_delta_rmse_vs_metadata": embedding["rmse"]
                - metadata["rmse"],
            }
        )
    return pd.DataFrame(rows)


def plot_results(
    null: pd.DataFrame,
    observed_r2: float,
    observed_rmse: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(null["r2"], bins=25, alpha=0.8)
    axes[0].axvline(observed_r2, color="red", linewidth=2, label="Correct pairs")
    axes[0].set(xlabel="LOSO R²", ylabel="Shuffles", title="Selection-aware R² null")
    axes[0].legend()
    axes[1].hist(null["rmse"], bins=25, alpha=0.8)
    axes[1].axvline(observed_rmse, color="red", linewidth=2, label="Correct pairs")
    axes[1].set(
        xlabel="LOSO RMSE", ylabel="Shuffles", title="Selection-aware RMSE null"
    )
    axes[1].legend()
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
        "inner_selection": f"{args.inner_site_folds}-fold site-grouped CV",
        "outer_evaluation": "leave-one-site-out",
        "nuisance_covariates": [
            "baseline_madrs",
            "log1p_baseline_window_count",
            "log1p_week8_window_count",
        ],
        "shuffles": args.shuffles,
        "shuffle_unit": "week-8 embeddings within site, with no fixed points",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; cached arrays were not loaded.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError(
                f"Existing run configuration differs: {config_path}. "
                "Use a different output directory."
            )
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    means, stds, recordings = load_recordings(input_dir, "train")
    baseline_means, week8_means, baseline_stds, week8_stds, pairs = make_pairs(
        means, stds, recordings
    )
    candidates = [
        Candidate(layer, pooling, alpha)
        for layer in args.layers
        for pooling in args.poolings
        for alpha in args.ridge_alphas
    ]
    observed_feature_sets = candidate_features(
        baseline_means,
        week8_means,
        baseline_stds,
        week8_stds,
        args.layers,
        args.poolings,
    )
    observed_embedding, observed_selections = evaluate_embedding_loso(
        observed_feature_sets,
        pairs,
        candidates,
        args.inner_site_folds,
        verbose=True,
    )
    covariates = recording_covariates(pairs)
    baseline_prediction = evaluate_linear_loso(covariates[:, :1], pairs)
    metadata_prediction = evaluate_linear_loso(covariates, pairs)
    truth = pairs["delta_madrs"].to_numpy(float)
    predictions = pairs[["subject_id", "site", "delta_madrs"]].copy()
    predictions["baseline_madrs"] = baseline_prediction
    predictions["baseline_plus_window_metadata"] = metadata_prediction
    predictions["baseline_plus_window_metadata_plus_embedding"] = observed_embedding
    predictions.to_csv(output_dir / "correct_pair_predictions.csv", index=False)

    null_path = output_dir / "selection_aware_null_distribution.csv"
    if null_path.exists():
        null = pd.read_csv(null_path)
        completed = set(null["shuffle"].astype(int))
        null_rows = null.to_dict("records")
        print(f"Resuming after {len(completed)} completed shuffles", flush=True)
    else:
        completed = set()
        null_rows = []
    sites = pairs["site"].to_numpy(str)
    for shuffle_index in range(1, args.shuffles + 1):
        if shuffle_index in completed:
            continue
        rng = np.random.default_rng(args.seed + shuffle_index)
        permutation = within_site_derangement(sites, rng)
        shuffled_feature_sets = candidate_features(
            baseline_means,
            week8_means[permutation],
            baseline_stds,
            week8_stds[permutation],
            args.layers,
            args.poolings,
        )
        shuffled_prediction, selections = evaluate_embedding_loso(
            shuffled_feature_sets,
            pairs,
            candidates,
            args.inner_site_folds,
            verbose=False,
        )
        values = metrics(truth, shuffled_prediction)
        null_rows.append(
            {
                "shuffle": shuffle_index,
                **values,
                "selection_counts": json.dumps(Counter(selections).most_common()),
            }
        )
        null = pd.DataFrame(null_rows).sort_values("shuffle")
        atomic_write_csv(null, null_path)
        print(
            f"completed selection-aware shuffle {shuffle_index}/{args.shuffles}: "
            f"r2={values['r2']:.3f} rmse={values['rmse']:.3f}",
            flush=True,
        )

    null = pd.DataFrame(null_rows).sort_values("shuffle").reset_index(drop=True)
    baseline_values = metrics(truth, baseline_prediction)
    metadata_values = metrics(truth, metadata_prediction)
    observed_values = metrics(truth, observed_embedding)
    bootstrap = paired_bootstrap(
        truth,
        metadata_prediction,
        observed_embedding,
        args.bootstrap_samples,
        args.seed,
    )
    summary = {
        "baseline_madrs": baseline_values,
        "baseline_plus_window_metadata": metadata_values,
        "baseline_plus_window_metadata_plus_embedding": observed_values,
        "embedding_delta_r2_vs_metadata": observed_values["r2"]
        - metadata_values["r2"],
        "embedding_delta_rmse_vs_metadata": observed_values["rmse"]
        - metadata_values["rmse"],
        "embedding_delta_r2_vs_metadata_ci": [
            bootstrap["r2_difference_ci_low"],
            bootstrap["r2_difference_ci_high"],
        ],
        "embedding_delta_rmse_vs_metadata_ci": [
            bootstrap["rmse_difference_ci_low"],
            bootstrap["rmse_difference_ci_high"],
        ],
        "selection_counts": Counter(observed_selections).most_common(),
        "null_r2_mean": float(null["r2"].mean()),
        "null_r2_2.5pct": float(null["r2"].quantile(0.025)),
        "null_r2_97.5pct": float(null["r2"].quantile(0.975)),
        "null_rmse_mean": float(null["rmse"].mean()),
        "null_rmse_2.5pct": float(null["rmse"].quantile(0.025)),
        "null_rmse_97.5pct": float(null["rmse"].quantile(0.975)),
        "selection_aware_permutation_p_r2": float(
            (1 + (null["r2"] >= observed_values["r2"]).sum()) / (len(null) + 1)
        ),
        "selection_aware_permutation_p_rmse": float(
            (1 + (null["rmse"] <= observed_values["rmse"]).sum())
            / (len(null) + 1)
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    per_site_metrics(
        pairs,
        baseline_prediction,
        metadata_prediction,
        observed_embedding,
    ).to_csv(output_dir / "per_site_metrics.csv", index=False)
    plot_results(
        null,
        observed_values["r2"],
        observed_values["rmse"],
        output_dir / "selection_aware_control.png",
    )

    lines = [
        "# Selection-aware WavLM longitudinal control",
        "",
        f"Subjects: **{len(pairs)}** across **{pairs['site'].nunique()}** sites. ",
        f"The null contains **{len(null)}** within-site week-8 derangements.",
        "Every observed and shuffled dataset repeats the full layer, pooling, and",
        "Ridge-alpha selection inside each outer leave-one-site-out fold.",
        "",
        "## Recording-quantity control",
        "",
        "| Model | R² | RMSE | Pearson |",
        "|---|---:|---:|---:|",
        f"| Baseline MADRS | {baseline_values['r2']:.3f} | "
        f"{baseline_values['rmse']:.3f} | {baseline_values['pearson']:.3f} |",
        f"| Baseline MADRS + window counts | {metadata_values['r2']:.3f} | "
        f"{metadata_values['rmse']:.3f} | {metadata_values['pearson']:.3f} |",
        f"| Baseline MADRS + window counts + WavLM change | "
        f"{observed_values['r2']:.3f} | {observed_values['rmse']:.3f} | "
        f"{observed_values['pearson']:.3f} |",
        "",
        f"WavLM versus metadata: ΔR² {summary['embedding_delta_r2_vs_metadata']:.3f} "
        f"[{bootstrap['r2_difference_ci_low']:.3f}, "
        f"{bootstrap['r2_difference_ci_high']:.3f}], ΔRMSE "
        f"{summary['embedding_delta_rmse_vs_metadata']:.3f} "
        f"[{bootstrap['rmse_difference_ci_low']:.3f}, "
        f"{bootstrap['rmse_difference_ci_high']:.3f}].",
        "",
        "## Selection-aware pairing null",
        "",
        f"Correct-pair R²: **{observed_values['r2']:.3f}**. Shuffled mean and "
        f"95% interval: {summary['null_r2_mean']:.3f} "
        f"[{summary['null_r2_2.5pct']:.3f}, {summary['null_r2_97.5pct']:.3f}].",
        f"Correct-pair RMSE: **{observed_values['rmse']:.3f}**. Shuffled mean and "
        f"95% interval: {summary['null_rmse_mean']:.3f} "
        f"[{summary['null_rmse_2.5pct']:.3f}, {summary['null_rmse_97.5pct']:.3f}].",
        f"Permutation p-values: R² **{summary['selection_aware_permutation_p_r2']:.4f}**, "
        f"RMSE **{summary['selection_aware_permutation_p_rmse']:.4f}**.",
        "",
        f"Correct-pair selections: `{json.dumps(Counter(observed_selections).most_common())}`.",
        "",
        "This test accounts for representation-selection optimism. Window counts",
        "control recording quantity, but not microphone, room, medication, or other",
        "visit-specific changes that could correlate with treatment response.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
