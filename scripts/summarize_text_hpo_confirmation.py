from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_PATTERN = re.compile(r"trial-(\d+)-seed-(\d+)$")
TRIAL_LABELS = {
    36: "HPO winner: rank 8, 25 s window",
    4: "Near-tie: rank 4, 20 s window",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize paired-seed MentalBERT HPO confirmation runs"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--screening-root", type=Path)
    return parser.parse_args()


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def summarize_run(run_dir: Path) -> dict[str, float | int | str] | None:
    match = RUN_PATTERN.fullmatch(run_dir.name)
    history_path = run_dir / "training_history.csv"
    arguments_path = run_dir / "training_arguments.json"
    complete_path = run_dir / ".complete"
    if (
        match is None
        or not complete_path.is_file()
        or not history_path.is_file()
        or not arguments_path.is_file()
    ):
        return None
    history = pd.read_csv(history_path)
    if history.empty:
        return None
    best = history.loc[history["validation_regression_r2"].idxmax()]
    best_epoch = int(best["epoch"])
    prediction_path = run_dir / f"recording_predictions_epoch_{best_epoch}.csv"
    diagnostics: dict[str, float] = {}
    if prediction_path.is_file():
        predictions = pd.read_csv(prediction_path)
        truth = predictions["regression_truth"].to_numpy(dtype=float)
        prediction = predictions["regression_prediction"].to_numpy(dtype=float)
        truth_std = float(np.std(truth, ddof=1))
        prediction_std = float(np.std(prediction, ddof=1))
        diagnostics = {
            "pearson": safe_correlation(truth, prediction),
            "prediction_target_sd_ratio": (
                prediction_std / truth_std if truth_std > 0 else float("nan")
            ),
            "mean_error": float(np.mean(prediction - truth)),
        }
    arguments = json.loads(arguments_path.read_text(encoding="utf-8"))
    source_trial = int(match.group(1))
    return {
        "source_trial": source_trial,
        "configuration": TRIAL_LABELS.get(source_trial, f"Trial {source_trial}"),
        "seed": int(match.group(2)),
        "best_epoch": best_epoch,
        "validation_r2": float(best["validation_regression_r2"]),
        "validation_rmse": float(best["validation_regression_rmse"]),
        "validation_rmse_original_scale": float(
            best["validation_regression_rmse_original_scale"]
        ),
        "learning_rate": float(arguments["learning_rate"]),
        "lora_rank": int(arguments["lora_rank"]),
        "lora_alpha": int(arguments["lora_alpha"]),
        "window_seconds": float(arguments["window_seconds"]),
        "stride_seconds": float(arguments["stride_seconds"]),
        "gradient_accumulation_steps": int(
            arguments["gradient_accumulation_steps"]
        ),
        **diagnostics,
    }


def screening_results(screening_root: Path | None) -> dict[int, float]:
    if screening_root is None:
        return {}
    results: dict[int, float] = {}
    for trial in TRIAL_LABELS:
        path = screening_root / f"trial-{trial:04d}" / "training_history.csv"
        if not path.is_file():
            continue
        history = pd.read_csv(path)
        if not history.empty:
            results[trial] = float(history["validation_regression_r2"].max())
    return results


def formatted(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def write_report(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    screening: dict[int, float],
    output_path: Path,
) -> None:
    lines = [
        "# MentalBERT HPO cross-seed confirmation",
        "",
        "The confirmation comparison uses new seeds 41, 42, and 43. Seed 40",
        "values are HPO screening references and are excluded from confirmation means.",
        "",
        "## Aggregate results",
        "",
        "| Trial | Configuration | Screening R2 | Seeds | Mean R2 | R2 SD | Min R2 | Max R2 | Mean original RMSE |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        screening_r2 = screening.get(int(row.source_trial), float("nan"))
        lines.append(
            f"| {row.source_trial} | {row.configuration} | "
            f"{formatted(screening_r2)} | {row.seeds} | "
            f"{formatted(row.mean_r2)} | {formatted(row.std_r2)} | "
            f"{formatted(row.min_r2)} | {formatted(row.max_r2)} | "
            f"{formatted(row.mean_original_rmse)} |"
        )

    lines.extend(
        [
            "",
            "## Individual runs",
            "",
            "| Trial | Seed | Best epoch | R2 | Original RMSE | Pearson r | Prediction/target SD | Mean error |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in runs.sort_values(["seed", "source_trial"]).itertuples(index=False):
        lines.append(
            f"| {row.source_trial} | {row.seed} | {row.best_epoch} | "
            f"{formatted(row.validation_r2)} | "
            f"{formatted(row.validation_rmse_original_scale)} | "
            f"{formatted(row.pearson)} | "
            f"{formatted(row.prediction_target_sd_ratio)} | "
            f"{formatted(row.mean_error)} |"
        )

    paired = runs.pivot(index="seed", columns="source_trial", values="validation_r2")
    lines.extend(["", "## Paired comparison", ""])
    if 36 in paired and 4 in paired:
        differences = (paired[36] - paired[4]).dropna()
        lines.append(
            f"Across {len(differences)} paired seeds, trial 36 minus trial 4 has "
            f"mean R2 difference {formatted(float(differences.mean()))}; trial 36 "
            f"wins {int((differences > 0).sum())} seeds and trial 4 wins "
            f"{int((differences < 0).sum())}."
        )
    else:
        lines.append("Paired results are incomplete.")
    lines.extend(
        [
            "",
            "Prefer the cross-seed mean over the original screening rank. Differences",
            "below roughly 0.02 R2 should be treated as inconclusive on this validation set.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root not found: {output_root}")
    rows = []
    for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed confirmation runs found in {output_root}")

    runs = pd.DataFrame(rows).sort_values(["source_trial", "seed"])
    runs.to_csv(output_root / "confirmation_runs.csv", index=False)
    summary = (
        runs.groupby(["source_trial", "configuration"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            mean_r2=("validation_r2", "mean"),
            std_r2=("validation_r2", "std"),
            min_r2=("validation_r2", "min"),
            max_r2=("validation_r2", "max"),
            mean_original_rmse=("validation_rmse_original_scale", "mean"),
        )
        .sort_values("mean_r2", ascending=False)
    )
    summary.to_csv(output_root / "confirmation_summary.csv", index=False)
    screening_root = (
        args.screening_root.expanduser().resolve()
        if args.screening_root is not None
        else None
    )
    report_path = output_root / "report.md"
    write_report(
        runs,
        summary,
        screening_results(screening_root),
        report_path,
    )
    print(summary.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
