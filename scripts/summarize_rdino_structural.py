from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_metrics import intraclass_correlation_2_1


TRAINABLE_PATTERN = re.compile(
    r"trainable parameters:\s*([\d,]+)\s*/\s*([\d,]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize audio-only RDINO structural experiments"
    )
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def parameter_counts(run_dir: Path) -> tuple[float, float]:
    log_path = run_dir / "training.log"
    if not log_path.is_file():
        return float("nan"), float("nan")
    match = TRAINABLE_PATTERN.search(log_path.read_text(encoding="utf-8"))
    if match is None:
        return float("nan"), float("nan")
    return (
        float(match.group(1).replace(",", "")),
        float(match.group(2).replace(",", "")),
    )


def summarize_run(run_dir: Path) -> dict | None:
    if not (run_dir / ".complete").is_file():
        return None
    history_path = run_dir / "training_history.csv"
    arguments_path = run_dir / "training_arguments.json"
    if not history_path.is_file() or not arguments_path.is_file():
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
        predicted = predictions["regression_prediction"].to_numpy(dtype=float)
        truth_std = float(np.std(truth, ddof=1))
        prediction_std = float(np.std(predicted, ddof=1))
        diagnostics = {
            "pearson": safe_correlation(truth, predicted),
            "icc_2_1": intraclass_correlation_2_1(truth, predicted),
            "prediction_target_sd_ratio": (
                prediction_std / truth_std if truth_std > 0 else float("nan")
            ),
            "mean_error": float(np.mean(predicted - truth)),
        }
    arguments = json.loads(arguments_path.read_text(encoding="utf-8"))
    trainable, total = parameter_counts(run_dir)
    return {
        "run": run_dir.name,
        "best_epoch": best_epoch,
        "validation_r2": float(best["validation_regression_r2"]),
        "validation_rmse": float(best["validation_regression_rmse"]),
        "validation_rmse_original_scale": float(
            best["validation_regression_rmse_original_scale"]
        ),
        "fusion_architecture": arguments.get("fusion_architecture", "two_layer"),
        "rdino_lora": not arguments["disable_rdino_lora"],
        "rdino_lora_scope": arguments.get("rdino_lora_scope", "terminal"),
        "lora_rank": int(arguments["lora_rank"]),
        "base_learning_rate": float(arguments["learning_rate"]),
        "audio_learning_rate": float(
            arguments.get("audio_learning_rate") or arguments["learning_rate"]
        ),
        "head_learning_rate": float(
            arguments.get("head_learning_rate") or arguments["learning_rate"]
        ),
        "trainable_parameters": trainable,
        "total_parameters": total,
        **diagnostics,
    }


def formatted(value, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    rows = []
    for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        row = summarize_run(run_dir)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed RDINO runs found in {output_root}")

    summary = pd.DataFrame(rows).sort_values("validation_r2", ascending=False)
    summary.to_csv(output_root / "summary.csv", index=False)
    columns = [
        ("run", "Run"),
        ("best_epoch", "Epoch"),
        ("validation_r2", "R2"),
        ("validation_rmse_original_scale", "RMSE"),
        ("icc_2_1", "ICC(2,1)"),
        ("pearson", "Pearson"),
        ("prediction_target_sd_ratio", "Pred/target SD"),
        ("trainable_parameters", "Trainable"),
    ]
    lines = [
        "# RDINO structural experiments",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| "
            + " | ".join(formatted(row.get(name, float("nan"))) for name, _ in columns)
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpret comparisons in order: run 00 tests the corrected LR/protocol;",
            "00 versus 01 isolates head depth; 01 versus 02 isolates discriminative",
            "learning rates; 02 versus 03 isolates adapter coverage; and 04 tests",
            "whether pretrained frozen RDINO embeddings already carry the signal.",
            "",
            "A low prediction/target SD ratio indicates regression toward the mean even",
            "when correlation is positive. Detailed values are in `summary.csv`.",
        ]
    )
    report_path = output_root / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
