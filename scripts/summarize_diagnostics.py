from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize structural diagnostic training runs"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def prediction_diagnostics(run_dir: Path, best_epoch: int) -> dict[str, float | int]:
    path = run_dir / f"recording_predictions_epoch_{best_epoch}.csv"
    if not path.is_file():
        return {}
    frame = pd.read_csv(path)
    truth = frame["regression_truth_standardized"].to_numpy(dtype=float)
    prediction = frame["regression_prediction_standardized"].to_numpy(dtype=float)
    truth_std = float(np.std(truth, ddof=1))
    prediction_std = float(np.std(prediction, ddof=1))
    class_truth = frame["class_truth"].to_numpy(dtype=int)
    class_prediction = frame["class_prediction"].to_numpy(dtype=int)
    matrix = confusion_matrix(class_truth, class_prediction, labels=[0, 1, 2, 3])
    class_counts = matrix.sum(axis=1)
    recalls = np.divide(
        np.diag(matrix),
        class_counts,
        out=np.zeros(4, dtype=float),
        where=class_counts > 0,
    )
    regression_classes = np.digitize(
        frame["regression_prediction"].to_numpy(dtype=float),
        [4.0, 11.0, 22.0],
        right=True,
    )
    result: dict[str, float | int] = {
        "recordings": len(frame),
        "regression_pearson": safe_correlation(truth, prediction),
        "prediction_target_sd_ratio": (
            prediction_std / truth_std if truth_std > 0 else float("nan")
        ),
        "head_disagreement_fraction": float(
            np.mean(class_prediction != regression_classes)
        ),
    }
    for class_index in range(4):
        result[f"class_{class_index}_recall"] = float(recalls[class_index])
        result[f"class_{class_index}_predictions"] = int(
            np.sum(class_prediction == class_index)
        )
    return result


def gradient_diagnostics(run_dir: Path) -> tuple[dict[str, float], pd.DataFrame]:
    path = run_dir / "gradient_diagnostics.csv"
    if not path.is_file():
        return {}, pd.DataFrame()
    frame = pd.read_csv(path)
    grouped_rows: list[dict[str, float | str | int]] = []
    for parameter_group, group in frame.groupby("parameter_group", sort=False):
        cosine = group["cosine_similarity"].to_numpy(dtype=float)
        ratio = group["weighted_class_to_regression_norm_ratio"].to_numpy(
            dtype=float
        )
        valid_cosine = cosine[np.isfinite(cosine)]
        valid_ratio = ratio[np.isfinite(ratio)]
        grouped_rows.append(
            {
                "parameter_group": parameter_group,
                "epochs_logged": len(group),
                "mean_cosine_similarity": (
                    float(np.mean(valid_cosine)) if len(valid_cosine) else float("nan")
                ),
                "gradient_conflict_fraction": (
                    float(np.mean(valid_cosine < 0)) if len(valid_cosine) else float("nan")
                ),
                "median_weighted_class_to_regression_norm_ratio": (
                    float(np.median(valid_ratio)) if len(valid_ratio) else float("nan")
                ),
            }
        )
    grouped = pd.DataFrame(grouped_rows)
    overall = grouped.loc[grouped["parameter_group"].eq("all_shared")]
    if overall.empty:
        return {}, grouped
    row = overall.iloc[0]
    return {
        "mean_gradient_cosine": float(row["mean_cosine_similarity"]),
        "gradient_conflict_fraction": float(row["gradient_conflict_fraction"]),
        "median_weighted_gradient_ratio": float(
            row["median_weighted_class_to_regression_norm_ratio"]
        ),
    }, grouped


def summarize_run(run_dir: Path) -> tuple[dict, pd.DataFrame] | None:
    history_path = run_dir / "training_history.csv"
    arguments_path = run_dir / "training_arguments.json"
    if not history_path.is_file() or not arguments_path.is_file():
        return None
    history = pd.read_csv(history_path)
    if history.empty:
        return None
    arguments = json.loads(arguments_path.read_text(encoding="utf-8"))
    best = history.loc[history["validation_regression_r2"].idxmax()]
    best_epoch = int(best["epoch"])
    gradient_summary, grouped_gradients = gradient_diagnostics(run_dir)
    row = {
        "run": run_dir.name,
        "modality": arguments["modality"],
        "text_model": arguments["text_model"],
        "normalization": arguments["embedding_normalization"],
        "sampling": arguments["train_sampling"],
        "text_lora": (
            arguments["modality"] in {"both", "text"}
            and not arguments["disable_text_lora"]
        ),
        "rdino_lora": (
            arguments["modality"] in {"both", "audio"}
            and not arguments["disable_rdino_lora"]
        ),
        "best_epoch": best_epoch,
        "validation_r2": float(best["validation_regression_r2"]),
        "validation_rmse": float(best["validation_regression_rmse"]),
        "balanced_accuracy": float(best["validation_balanced_accuracy"]),
        "macro_f1": float(best["validation_macro_f1"]),
        **prediction_diagnostics(run_dir, best_epoch),
        **gradient_summary,
    }
    if not grouped_gradients.empty:
        grouped_gradients.insert(0, "run", run_dir.name)
    return row, grouped_gradients


def format_value(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    rows: list[dict] = []
    gradient_frames: list[pd.DataFrame] = []
    for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        result = summarize_run(run_dir)
        if result is None:
            continue
        row, gradients = result
        rows.append(row)
        if not gradients.empty:
            gradient_frames.append(gradients)
    if not rows:
        raise RuntimeError(f"No completed diagnostic runs found in {output_root}")

    summary = pd.DataFrame(rows).sort_values("run")
    summary.to_csv(output_root / "summary.csv", index=False)
    if gradient_frames:
        pd.concat(gradient_frames, ignore_index=True).to_csv(
            output_root / "gradient_summary_by_group.csv", index=False
        )

    columns = [
        ("run", "Run"),
        ("validation_r2", "R²"),
        ("validation_rmse", "RMSE"),
        ("balanced_accuracy", "BA"),
        ("prediction_target_sd_ratio", "Pred/target SD"),
        ("class_0_recall", "Recall 0"),
        ("class_1_recall", "Recall 1"),
        ("mean_gradient_cosine", "Grad cosine"),
        ("gradient_conflict_fraction", "Conflict frac."),
        ("median_weighted_gradient_ratio", "Weighted C/R norm"),
    ]
    lines = [
        "# Structural diagnostic experiments",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                format_value(row.get(name, float("nan"))) for name, _ in columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Gradient cosine below zero indicates conflicting task directions. The weighted",
            "classification/regression norm ratio includes the configured task weights; values",
            "far above one mean classification dominates the shared update.",
            "",
            "Detailed results are in `summary.csv` and `gradient_summary_by_group.csv`.",
        ]
    )
    (output_root / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Wrote {output_root / 'report.md'}")


if __name__ == "__main__":
    main()
