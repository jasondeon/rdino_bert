"""Score complete LOSO predictions on the frozen common recording cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_evaluation import regression_metrics


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_bundle(bundle):
    for name, expected in json.loads((bundle / "checksums.json").read_text()).items():
        actual = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")


def collect_predictions(bundle, prediction_rows, level, window_view="all"):
    records = read_csv(bundle / "recordings.csv")
    key = "recording_id" if level == "recording" else "window_id"
    units = records if level == "recording" else read_csv(bundle / "windows.csv")
    if level == "window" and window_view == "text":
        units = [r for r in units if int(r["word_count"]) > 0]
    expected = {r[key]: r for r in units}
    studies = {r["recording_id"]: r["study"] for r in records}
    found = {}
    for row in prediction_rows:
        unit_id = row[key]
        if unit_id not in expected:
            raise ValueError(f"Unexpected {key}: {unit_id}")
        if unit_id in found:
            raise ValueError(f"Duplicate prediction for {unit_id}")
        recording_id = expected[unit_id]["recording_id"]
        if row["test_study"] != studies[recording_id]:
            raise ValueError("Prediction test_study does not match recording's outer holdout")
        value = float(row["prediction"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite prediction")
        found[unit_id] = value
    missing = set(expected) - set(found)
    if missing:
        raise ValueError(f"Missing {len(missing)} predictions; no silent cohort changes")
    totals = defaultdict(float)
    weights = defaultdict(float)
    for unit_id, unit in expected.items():
        weight = 1.0 if level == "recording" else float(unit["aggregation_weight"])
        totals[unit["recording_id"]] += weight * found[unit_id]
        weights[unit["recording_id"]] += weight
    if set(weights) != set(studies):
        raise ValueError("Selected view does not cover the complete recording cohort")
    for recording_id, weight in weights.items():
        if weight <= 0 or (window_view == "all" and not math.isclose(weight, 1.0, abs_tol=1e-8)):
            raise ValueError(f"Aggregation weights do not sum to one: {recording_id}")
    return records, {k: v / weights[k] for k, v in totals.items()}


def evaluate(bundle, prediction_rows, level="recording", window_view="all"):
    verify_bundle(bundle)
    records, predictions = collect_predictions(bundle, prediction_rows, level, window_view)
    by_study = []
    baselines = {}
    for study in sorted({r["study"] for r in records}):
        train = read_csv(bundle / "loso" / study / "train.csv")
        test = [r for r in records if r["study"] == study]
        train_subjects = {r["subject_id"] for r in train}
        if train_subjects & {r["subject_id"] for r in test}:
            raise ValueError("Subject leakage in bundle")
        mean = sum(float(r["regression_label"]) for r in train) / len(train)
        labels = [float(r["regression_label"]) for r in test]
        predicted = [predictions[r["recording_id"]] for r in test]
        baselines.update({r["recording_id"]: mean for r in test})
        by_study.append({
            "study": study, "subjects": len({r["subject_id"] for r in test}),
            "model": regression_metrics(labels, predicted),
            "training_mean_baseline": regression_metrics(labels, [mean] * len(test)),
        })
    labels = [float(r["regression_label"]) for r in records]
    predicted = [predictions[r["recording_id"]] for r in records]
    subject_errors = defaultdict(list)
    for record, target, prediction in zip(records, labels, predicted):
        subject_errors[record["subject_id"]].append(abs(prediction - target))
    report = {
        "prediction_level": level,
        "window_view": window_view,
        "recordings": len(records), "subjects": len(subject_errors),
        "by_study": by_study,
        "macro_study": {
            name: sum(s["model"][name] for s in by_study) / len(by_study)
            for name in ("mae", "rmse")
        },
        "macro_study_training_mean_baseline": {
            name: sum(s["training_mean_baseline"][name] for s in by_study) / len(by_study)
            for name in ("mae", "rmse")
        },
        "pooled": regression_metrics(labels, predicted),
        "pooled_training_mean_baseline": regression_metrics(labels, [baselines[r["recording_id"]] for r in records]),
        "equal_subject_mae": sum(sum(v) / len(v) for v in subject_errors.values()) / len(subject_errors),
        "notes": [
            "MAE/RMSE are in original MADRS points. Mean error is prediction minus truth.",
            "Calibration is truth = intercept + slope * prediction; diagnostic only, never applied to test predictions.",
            "Pooled R2 includes between-study variation; inspect every study separately.",
            "Null correlations/R2/slopes indicate undefined values, not zero performance.",
            "test_study and coverage are checked; this cannot prove how an external model was trained.",
        ],
    }
    output = [{
        "recording_id": r["recording_id"], "subject_id": r["subject_id"],
        "study": r["study"], "truth": float(r["regression_label"]),
        "prediction": predictions[r["recording_id"]],
        "training_mean_baseline": baselines[r["recording_id"]],
    } for r in records]
    return report, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--level", choices=("recording", "window"), default="recording")
    parser.add_argument("--window-view", choices=("all", "text"), default="all",
                        help="All audio windows, or predefined nonempty-text windows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new directory")
    if args.level == "recording" and args.window_view != "all":
        parser.error("--window-view applies only to window predictions")
    report, records = evaluate(args.bundle, read_csv(args.predictions), args.level, args.window_view)
    report["predictions_sha256"] = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    report["bundle_checksums_sha256"] = hashlib.sha256((args.bundle / "checksums.json").read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "recording_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"macro_study": report["macro_study"], "pooled": report["pooled"]}, indent=2))


if __name__ == "__main__":
    main()
