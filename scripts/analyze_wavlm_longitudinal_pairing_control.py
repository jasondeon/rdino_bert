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
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_wavlm_domain_generalization import load_recordings
from analyze_wavlm_longitudinal_change import (
    baseline_predict,
    combined_predict,
    make_pairs,
    metrics,
    ridge_predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare correctly paired longitudinal WavLM changes with "
            "within-site shuffled week-8 negative controls"
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--shuffles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_features(
    baseline_means: np.ndarray,
    week8_means: np.ndarray,
    baseline_stds: np.ndarray,
    week8_stds: np.ndarray,
    layer: int,
) -> np.ndarray:
    return np.concatenate(
        [
            week8_means[:, layer] - baseline_means[:, layer],
            week8_stds[:, layer] - baseline_stds[:, layer],
        ],
        axis=1,
    ).astype(np.float64, copy=False)


def within_site_derangement(sites: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permutation = np.arange(len(sites))
    for site in np.unique(sites):
        indices = np.flatnonzero(sites == site)
        if len(indices) < 2:
            raise ValueError(f"Site {site} has only one paired subject")
        for _ in range(1000):
            shuffled = rng.permutation(indices)
            if np.all(shuffled != indices):
                permutation[indices] = shuffled
                break
        else:
            raise RuntimeError(f"Could not construct a derangement for site {site}")
    return permutation


def evaluate_loso(
    features: np.ndarray,
    pairs: pd.DataFrame,
    alpha: float,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    truth = pairs["delta_madrs"].to_numpy(float)
    baseline = pairs["baseline_madrs"].to_numpy(float)
    sites = pairs["site"].to_numpy(str)
    predictions = {
        "baseline_madrs": np.full(len(pairs), np.nan),
        "delta_embedding": np.full(len(pairs), np.nan),
        "baseline_plus_delta_embedding": np.full(len(pairs), np.nan),
    }
    splitter = GroupKFold(n_splits=len(np.unique(sites)))
    for fit, held_out in splitter.split(features, groups=sites):
        predictions["baseline_madrs"][held_out] = baseline_predict(
            baseline[fit], truth[fit], baseline[held_out]
        )
        predictions["delta_embedding"][held_out] = ridge_predict(
            features[fit], truth[fit], features[held_out], alpha
        )
        predictions["baseline_plus_delta_embedding"][held_out] = combined_predict(
            features[fit],
            baseline[fit],
            truth[fit],
            features[held_out],
            baseline[held_out],
            alpha,
        )
    frame = pairs[["subject_id", "site", "delta_madrs"]].copy()
    result = {}
    for name, prediction in predictions.items():
        if np.isnan(prediction).any():
            raise RuntimeError(f"Missing predictions for {name}")
        frame[name] = prediction
        result[name] = metrics(truth, prediction)
    return result, frame


def permutation_summary(
    observed: dict[str, dict[str, float]], null: pd.DataFrame
) -> pd.DataFrame:
    baseline_r2 = observed["baseline_madrs"]["r2"]
    baseline_rmse = observed["baseline_madrs"]["rmse"]
    rows = []
    for model in ("delta_embedding", "baseline_plus_delta_embedding"):
        observed_r2 = observed[model]["r2"]
        observed_rmse = observed[model]["rmse"]
        null_r2 = null[f"{model}_r2"].to_numpy(float)
        null_rmse = null[f"{model}_rmse"].to_numpy(float)
        rows.append(
            {
                "model": model,
                "observed_r2": observed_r2,
                "observed_rmse": observed_rmse,
                "observed_delta_r2_vs_baseline": observed_r2 - baseline_r2,
                "observed_delta_rmse_vs_baseline": observed_rmse - baseline_rmse,
                "null_r2_mean": float(np.mean(null_r2)),
                "null_r2_2.5pct": float(np.percentile(null_r2, 2.5)),
                "null_r2_97.5pct": float(np.percentile(null_r2, 97.5)),
                "null_rmse_mean": float(np.mean(null_rmse)),
                "null_rmse_2.5pct": float(np.percentile(null_rmse, 2.5)),
                "null_rmse_97.5pct": float(np.percentile(null_rmse, 97.5)),
                "permutation_p_r2": float(
                    (1 + np.sum(null_r2 >= observed_r2)) / (len(null_r2) + 1)
                ),
                "permutation_p_rmse": float(
                    (1 + np.sum(null_rmse <= observed_rmse)) / (len(null_rmse) + 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_null(
    summary: pd.DataFrame, null: pd.DataFrame, output_path: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    models = ("delta_embedding", "baseline_plus_delta_embedding")
    for column, model in enumerate(models):
        row = summary.set_index("model").loc[model]
        axes[0, column].hist(null[f"{model}_r2"], bins=30, alpha=0.8)
        axes[0, column].axvline(row["observed_r2"], color="red", linewidth=2)
        axes[0, column].set(
            title=model.replace("_", " ").title(), xlabel="LOSO R²", ylabel="Shuffles"
        )
        axes[1, column].hist(null[f"{model}_rmse"], bins=30, alpha=0.8)
        axes[1, column].axvline(row["observed_rmse"], color="red", linewidth=2)
        axes[1, column].set(xlabel="LOSO RMSE", ylabel="Shuffles")
    figure.suptitle("Within-site shuffled week-8 null distributions (red = correct pairs)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extraction = json.loads((input_dir / "extraction_config.json").read_text())
    layer_names = list(extraction["layer_names"])
    if args.layer < 0 or args.layer >= len(layer_names):
        raise ValueError(f"Layer must be between 0 and {len(layer_names) - 1}")
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": extraction["model_name"],
        "window_seconds": extraction["window_seconds"],
        "stride_seconds": extraction["stride_seconds"],
        "layer": args.layer,
        "layer_name": layer_names[args.layer],
        "pooling": "mean_temporal_std",
        "ridge_alpha": args.ridge_alpha,
        "shuffles": args.shuffles,
        "shuffle_unit": "week-8 embeddings within site, with no fixed points",
        "evaluation": "leave-one-site-out",
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
    observed_features = make_features(
        baseline_means,
        week8_means,
        baseline_stds,
        week8_stds,
        args.layer,
    )
    observed, observed_predictions = evaluate_loso(
        observed_features, pairs, args.ridge_alpha
    )

    rng = np.random.default_rng(args.seed)
    null_rows = []
    sites = pairs["site"].to_numpy(str)
    for shuffle_index in range(args.shuffles):
        permutation = within_site_derangement(sites, rng)
        shuffled_features = make_features(
            baseline_means,
            week8_means[permutation],
            baseline_stds,
            week8_stds[permutation],
            args.layer,
        )
        shuffled, _ = evaluate_loso(shuffled_features, pairs, args.ridge_alpha)
        null_rows.append(
            {
                "shuffle": shuffle_index + 1,
                **{
                    f"{model}_{metric}": values[metric]
                    for model, values in shuffled.items()
                    if model != "baseline_madrs"
                    for metric in ("r2", "rmse", "pearson", "spearman")
                },
            }
        )
        if (shuffle_index + 1) % 25 == 0 or shuffle_index + 1 == args.shuffles:
            print(f"completed shuffles: {shuffle_index + 1}/{args.shuffles}", flush=True)
    null = pd.DataFrame(null_rows)
    summary = permutation_summary(observed, null)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    observed_predictions.to_csv(output_dir / "correct_pair_predictions.csv", index=False)
    null.to_csv(output_dir / "shuffled_null_distribution.csv", index=False)
    summary.to_csv(output_dir / "pairing_control_summary.csv", index=False)
    plot_null(summary, null, output_dir / "pairing_control.png")

    baseline = observed["baseline_madrs"]
    lines = [
        "# WavLM longitudinal pairing negative control",
        "",
        f"Correctly paired CBN17 subjects: **{len(pairs)}** across "
        f"**{pairs['site'].nunique()}** sites.",
        f"Representation: layer {args.layer} ({layer_names[args.layer]}), "
        f"mean + temporal-SD pooling, Ridge alpha {args.ridge_alpha:g}.",
        f"The null distribution contains {args.shuffles} within-site derangements "
        "of week-8 embeddings. MADRS outcomes and baseline embeddings remain attached "
        "to the correct subject.",
        "",
        f"Baseline-MADRS model: R² {baseline['r2']:.3f}, RMSE {baseline['rmse']:.3f}.",
        "",
        "| Model | Correct-pair R² | Shuffled R² mean [95% interval] | p(R²) | Correct-pair RMSE | Shuffled RMSE mean [95% interval] | p(RMSE) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['observed_r2']:.3f} | "
            f"{row['null_r2_mean']:.3f} [{row['null_r2_2.5pct']:.3f}, "
            f"{row['null_r2_97.5pct']:.3f}] | {row['permutation_p_r2']:.4f} | "
            f"{row['observed_rmse']:.3f} | {row['null_rmse_mean']:.3f} "
            f"[{row['null_rmse_2.5pct']:.3f}, {row['null_rmse_97.5pct']:.3f}] | "
            f"{row['permutation_p_rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "A small permutation p-value means correct subject pairing performs better",
            "than would be expected from site and visit distributions alone. It does not",
            "by itself establish that the signal is specifically caused by MADRS change.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
