from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


PROBES = ("mean", "mean_std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen RDINO embedding audits across window sizes"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/rdino-window-size-audit")
    )
    parser.add_argument(
        "--reference-20-dir",
        type=Path,
        default=Path("outputs/rdino-embedding-audit-preserved-gaps"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=40)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def audit_directories(output_root: Path, reference_20_dir: Path) -> dict[float, Path]:
    return {
        5.0: output_root / "window-05",
        10.0: output_root / "window-10",
        20.0: reference_20_dir,
        30.0: output_root / "window-30",
    }


def require_complete(directory: Path) -> None:
    required = (
        ".complete",
        "extraction_config.json",
        "probe_summary.csv",
        "embedding_similarity.csv",
        "mean_predictions.csv",
        "mean_std_predictions.csv",
    )
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete audit directory {directory}; missing: {', '.join(missing)}"
        )


def paired_bootstrap_r2_difference(
    truth: np.ndarray,
    prediction: np.ndarray,
    reference_prediction: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = float(
        r2_score(truth, prediction) - r2_score(truth, reference_prediction)
    )
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(truth), len(truth))
        sampled_truth = truth[indices]
        if np.var(sampled_truth) <= 0:
            continue
        differences.append(
            r2_score(sampled_truth, prediction[indices])
            - r2_score(sampled_truth, reference_prediction[indices])
        )
    if not differences:
        raise ValueError("Could not form any non-degenerate bootstrap samples")
    low, high = np.percentile(differences, [2.5, 97.5])
    return observed, float(low), float(high)


def aligned_predictions(
    directory: Path, reference_directory: Path, probe: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = ["recording_index", "audio_path", "subject_id"]
    current = pd.read_csv(
        directory / f"{probe}_predictions.csv", dtype={"subject_id": str}
    )
    reference = pd.read_csv(
        reference_directory / f"{probe}_predictions.csv", dtype={"subject_id": str}
    )
    columns = keys + ["regression_truth", "regression_prediction"]
    joined = current[columns].merge(
        reference[columns],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_reference"),
    )
    if len(joined) != len(current) or len(joined) != len(reference):
        raise ValueError(
            f"Validation recording cohort differs between {directory} and "
            f"{reference_directory} for probe {probe}"
        )
    truth = joined["regression_truth"].to_numpy(dtype=float)
    reference_truth = joined["regression_truth_reference"].to_numpy(dtype=float)
    if not np.allclose(truth, reference_truth):
        raise ValueError(f"Validation labels differ for probe {probe}")
    return (
        truth,
        joined["regression_prediction"].to_numpy(dtype=float),
        joined["regression_prediction_reference"].to_numpy(dtype=float),
    )


def load_results(
    directories: dict[float, Path], bootstrap_samples: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_directory = directories[20.0]
    configs: dict[float, dict] = {}
    probe_frames = []
    similarity_frames = []
    for expected_window, directory in directories.items():
        require_complete(directory)
        config = json.loads((directory / "extraction_config.json").read_text())
        actual_window = float(config["window_seconds"])
        if not np.isclose(actual_window, expected_window):
            raise ValueError(
                f"Expected {expected_window:g}s in {directory}, found {actual_window:g}s"
            )
        configs[expected_window] = config

        probes = pd.read_csv(directory / "probe_summary.csv")
        probes.insert(0, "window_seconds", expected_window)
        probes.insert(1, "stride_seconds", float(config["stride_seconds"]))
        probes.insert(
            2,
            "eligibility_window_seconds",
            float(config["eligibility_window_seconds"]),
        )
        probe_frames.append(probes)

        similarity = pd.read_csv(directory / "embedding_similarity.csv")
        similarity.insert(0, "window_seconds", expected_window)
        similarity.insert(1, "stride_seconds", float(config["stride_seconds"]))
        similarity_frames.append(similarity)

    reference = configs[20.0]
    invariants = ("train", "validation", "eligibility_window_seconds")
    for window, config in configs.items():
        for field in invariants:
            if config[field] != reference[field]:
                raise ValueError(
                    f"Audit configuration mismatch at {window:g}s for {field}: "
                    f"{config[field]!r} != {reference[field]!r}"
                )
        if config.get("speaker_gap_policy") != "preserve":
            raise ValueError(f"Audit at {window:g}s did not preserve unlabeled gaps")
        for split in ("train", "validation"):
            if config["splits"][split]["recordings"] != reference["splits"][split][
                "recordings"
            ]:
                raise ValueError(
                    f"Recording cohort differs at {window:g}s for split {split}"
                )

    summary = pd.concat(probe_frames, ignore_index=True)
    summary["delta_r2_vs_20"] = np.nan
    summary["delta_r2_vs_20_ci_low"] = np.nan
    summary["delta_r2_vs_20_ci_high"] = np.nan
    for index, row in summary.iterrows():
        probe = str(row["probe"])
        window = float(row["window_seconds"])
        if probe not in PROBES:
            continue
        if np.isclose(window, 20.0):
            summary.loc[index, [
                "delta_r2_vs_20",
                "delta_r2_vs_20_ci_low",
                "delta_r2_vs_20_ci_high",
            ]] = 0.0
            continue
        truth, prediction, reference_prediction = aligned_predictions(
            directories[window], reference_directory, probe
        )
        delta, low, high = paired_bootstrap_r2_difference(
            truth,
            prediction,
            reference_prediction,
            samples=bootstrap_samples,
            seed=seed + int(window * 10) + PROBES.index(probe),
        )
        summary.loc[index, "delta_r2_vs_20"] = delta
        summary.loc[index, "delta_r2_vs_20_ci_low"] = low
        summary.loc[index, "delta_r2_vs_20_ci_high"] = high
    return summary, pd.concat(similarity_frames, ignore_index=True)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"mean": "tab:blue", "mean_std": "tab:orange"}
    labels = {"mean": "Mean", "mean_std": "Mean + window SD"}
    for probe in PROBES:
        rows = summary[summary["probe"] == probe].sort_values("window_seconds")
        x = rows["window_seconds"].to_numpy(dtype=float)
        r2 = rows["r2"].to_numpy(dtype=float)
        r2_errors = np.maximum(
            0.0,
            np.vstack(
                [
                    r2 - rows["r2_ci_low"].to_numpy(dtype=float),
                    rows["r2_ci_high"].to_numpy(dtype=float) - r2,
                ]
            ),
        )
        axes[0].errorbar(
            x,
            r2,
            yerr=r2_errors,
            marker="o",
            capsize=4,
            color=colors[probe],
            label=labels[probe],
        )
        delta = rows["delta_r2_vs_20"].to_numpy(dtype=float)
        delta_errors = np.maximum(
            0.0,
            np.vstack(
                [
                    delta - rows["delta_r2_vs_20_ci_low"].to_numpy(dtype=float),
                    rows["delta_r2_vs_20_ci_high"].to_numpy(dtype=float) - delta,
                ]
            ),
        )
        axes[1].errorbar(
            x,
            delta,
            yerr=delta_errors,
            marker="o",
            capsize=4,
            color=colors[probe],
            label=labels[probe],
        )
    axes[0].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[0].set(
        title="Frozen-embedding validation signal",
        xlabel="Window size (seconds)",
        ylabel="Validation R² (95% bootstrap CI)",
    )
    axes[1].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[1].set(
        title="Paired difference from 20 seconds",
        xlabel="Window size (seconds)",
        ylabel="Δ validation R² (95% paired CI)",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xticks([5, 10, 20, 30])
        axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

def write_report(summary: pd.DataFrame, similarity: pd.DataFrame, path: Path) -> None:
    primary = summary[summary["probe"].isin(PROBES)].sort_values(
        ["window_seconds", "probe"]
    )
    best = primary.loc[primary["r2"].idxmax()]
    decisive_better = primary[
        (primary["window_seconds"] != 20)
        & (primary["delta_r2_vs_20_ci_low"] > 0)
    ]
    decisive_worse = primary[
        (primary["window_seconds"] != 20)
        & (primary["delta_r2_vs_20_ci_high"] < 0)
    ]
    if not decisive_better.empty:
        interpretation = (
            "At least one alternative has a paired R² interval entirely above the "
            "20-second result, providing evidence that window duration changes the "
            "usable frozen-RDINO signal."
        )
    elif not decisive_worse.empty:
        interpretation = (
            "No alternative is clearly better than 20 seconds, and at least one is "
            "clearly worse by paired bootstrap."
        )
    else:
        interpretation = (
            "The paired intervals do not distinguish the tested window sizes from "
            "20 seconds. Treat the apparent ranking as noise unless it replicates."
        )

    lines = [
        "# RDINO window-size embedding audit",
        "",
        "All audits use the same subject-disjoint manifests, recordings eligible for",
        "a 30-second window, preserved unlabeled gaps, and frozen RDINO checkpoint.",
        "Ridge regularization is selected with subject-grouped cross-validation on",
        "training recordings. Delta intervals resample validation recordings in pairs.",
        "",
        "## Probe results",
        "",
        "| Window | Stride | Probe | R² [95% CI] | ΔR² vs 20s [95% CI] | RMSE | ICC(2,1) | Pearson | Pred/target SD | Alpha |",
        "| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary.to_dict(orient="records"):
        lines.append(
            f"| {row['window_seconds']:g}s | {row['stride_seconds']:g}s | "
            f"{row['probe']} | {row['r2']:.4f} "
            f"[{row['r2_ci_low']:.4f}, {row['r2_ci_high']:.4f}] | "
            f"{row['delta_r2_vs_20']:.4f} "
            f"[{row['delta_r2_vs_20_ci_low']:.4f}, "
            f"{row['delta_r2_vs_20_ci_high']:.4f}] | {row['rmse']:.4f} | "
            f"{row['icc_2_1']:.4f} | {row['pearson']:.4f} | "
            f"{row['prediction_target_sd_ratio']:.4f} | {row['best_alpha']:.4g} |"
        )
    baseline = summary[
        (summary["window_seconds"] == 20) & (summary["probe"] == "training_mean")
    ].iloc[0]
    lines.extend(
        [
            "",
            f"Training-mean baseline: R² {baseline['r2']:.4f}, RMSE "
            f"{baseline['rmse']:.4f}.",
            "",
            "## Embedding geometry",
            "",
            "| Window | Split | Windows | Windows/recording | Within-recording cosine | Between-recording cosine | Between p95 | PCs for 90% variance |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in similarity.sort_values(["window_seconds", "split"]).to_dict(
        orient="records"
    ):
        lines.append(
            f"| {row['window_seconds']:g}s | {row['split']} | "
            f"{int(row['windows'])} | {row['mean_windows_per_recording']:.2f} | "
            f"{row['mean_window_to_recording_centroid_cosine']:.4f} | "
            f"{row['mean_between_recording_centroid_cosine']:.4f} | "
            f"{row['p95_between_recording_centroid_cosine']:.4f} | "
            f"{int(row['principal_components_for_90_percent_variance'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "",
            f"The largest point estimate is {best['r2']:.4f} for the "
            f"{best['probe']} probe at {best['window_seconds']:g} seconds. A larger "
            "point estimate alone is not evidence of improvement; use the paired",
            "interval and geometry changes to decide whether a structural follow-up is",
            "warranted.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output_root = resolve(args.output_root)
    reference_20_dir = resolve(args.reference_20_dir)
    directories = audit_directories(output_root, reference_20_dir)
    summary, similarity = load_results(
        directories, args.bootstrap_samples, args.seed
    )
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "window_size_summary.csv", index=False)
    similarity.to_csv(output_root / "window_size_geometry.csv", index=False)
    plot_summary(summary, output_root / "window_size_audit.png")
    write_report(summary, similarity, output_root / "report.md")
    print(summary.to_string(index=False))
    print(f"Wrote {output_root / 'report.md'}")


if __name__ == "__main__":
    main()
