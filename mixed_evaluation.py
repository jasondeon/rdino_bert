"""Validation and scoring for the frozen mixed-study subject folds."""
from __future__ import annotations

from collections import defaultdict
import math

from clinical_evaluation import regression_metrics
from scripts.score_evaluation_bundle import collect_predictions, read_csv, verify_bundle


def mixed_partitions(bundle):
    records = read_csv(bundle / "recordings.csv")
    canonical = {r["recording_id"]: r for r in records}
    if len(canonical) != len(records):
        raise ValueError("Duplicate recording IDs in bundle")
    studies = {r["study"] for r in records}
    assignment, subject_assignment, folds = {}, {}, {}

    def checked(path):
        rows = read_csv(path)
        ids = [r["recording_id"] for r in rows]
        if not rows or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicated partition: {path}")
        if any(canonical.get(r["recording_id"]) != r for r in rows):
            raise ValueError(f"Partition differs from canonical records: {path}")
        if {r["study"] for r in rows} != studies:
            raise ValueError(f"Every study must be represented in this mixed partition: {path}")
        return rows

    def check_pair(train, test, expected):
        if {r["subject_id"] for r in train} & {r["subject_id"] for r in test}:
            raise ValueError("Subject leakage in mixed folds")
        if {r["recording_id"] for r in train + test} != expected:
            raise ValueError("Mixed partitions do not cover the expected cohort")

    for directory in sorted((bundle / "mixed_subject").iterdir()):
        train, test = checked(directory / "train.csv"), checked(directory / "test.csv")
        check_pair(train, test, set(canonical))
        for row in test:
            recording_id, subject = row["recording_id"], row["subject_id"]
            if recording_id in assignment:
                raise ValueError("Recording appears in multiple outer test folds")
            if subject in subject_assignment and subject_assignment[subject] != directory.name:
                raise ValueError("Subject visits are split across outer test folds")
            assignment[recording_id] = directory.name
            subject_assignment[subject] = directory.name
        inner, validated = {}, []
        for inner_dir in sorted((directory / "inner").iterdir()):
            fit, validation = checked(inner_dir / "train.csv"), checked(inner_dir / "validation.csv")
            check_pair(fit, validation, {r["recording_id"] for r in train})
            inner[inner_dir.name] = {"train": fit, "validation": validation}
            validated.extend(r["recording_id"] for r in validation)
        if sorted(validated) != sorted(r["recording_id"] for r in train):
            raise ValueError("Each outer-training recording must be inner validation exactly once")
        folds[directory.name] = {"train": train, "test": test, "inner": inner}
    if set(assignment) != set(canonical):
        raise ValueError("Outer test folds do not cover every recording exactly once")
    return records, assignment, folds


def study_metrics(records, predictions):
    values = defaultdict(lambda: ([], []))
    for row, prediction in zip(records, predictions, strict=True):
        y, p = values[row["study"]]
        y.append(float(row["regression_label"]))
        p.append(float(prediction))
    return {study: regression_metrics(y, p) for study, (y, p) in sorted(values.items())}


def select_mixed_epoch(histories):
    """Pool inner OOF squared errors within study; then average study RMSEs.

    Inner subject folds each contain every study. Averaging their pooled RMSE
    would implicitly give larger studies more influence than in LOSO selection.
    """
    maps = [{int(row["epoch"]): row["by_study"] for row in history} for history in histories]
    if not maps or any(not m for m in maps):
        raise ValueError("Missing inner histories")
    epochs = sorted(set(maps[0]).intersection(*(set(m) for m in maps[1:])))
    if not epochs:
        raise ValueError("No common validation epochs")
    studies = set(maps[0][epochs[0]])
    curve = []
    for epoch in epochs:
        if any(set(m[epoch]) != studies for m in maps):
            raise ValueError("Inner validation study coverage differs")
        per_study = {}
        for study in sorted(studies):
            count = sum(m[epoch][study]["recordings"] for m in maps)
            sse = sum(m[epoch][study]["recordings"] * m[epoch][study]["rmse"]**2 for m in maps)
            per_study[study] = math.sqrt(sse / count)
        score = sum(per_study.values()) / len(per_study)
        if not math.isfinite(score):
            raise ValueError("Nonfinite epoch-selection score")
        curve.append({"epoch": epoch, "macro_study_rmse": score, "study_rmse": per_study})
    return min(curve, key=lambda row: (row["macro_study_rmse"], row["epoch"]))["epoch"], curve


def evaluate_mixed(bundle, rows, level="window", window_view="text"):
    verify_bundle(bundle)
    records, assignment, folds = mixed_partitions(bundle)
    canonical = {r["recording_id"]: r for r in records}
    if level not in {"recording", "window"} or window_view not in {"all", "text"}:
        raise ValueError("Unknown prediction level or window view")
    window_ids = {w["window_id"]: w["recording_id"] for w in read_csv(bundle / "windows.csv")}
    converted = []
    for row in rows:
        recording_id = row.get("recording_id") if level == "recording" else window_ids.get(row.get("window_id"))
        if recording_id not in canonical:
            raise ValueError("Unknown recording or window prediction")
        if row.get("test_fold") != assignment[recording_id]:
            raise ValueError("Prediction belongs to the wrong mixed outer test fold")
        if row.get("study") != canonical[recording_id]["study"]:
            raise ValueError("Prediction study does not match source record")
        # Reuse the unchanged canonical window-coverage and aggregation checks.
        converted.append({**row, "test_study": row["study"]})
    _, prediction = collect_predictions(bundle, converted, level, window_view)
    global_baseline, study_baseline = {}, {}
    by_fold = []
    for name, fold in folds.items():
        train, test = fold["train"], fold["test"]
        mean = sum(float(r["regression_label"]) for r in train) / len(train)
        for study in {r["study"] for r in train}:
            group = [r for r in train if r["study"] == study]
            study_mean = sum(float(r["regression_label"]) for r in group) / len(group)
            for row in test:
                if row["study"] == study:
                    study_baseline[row["recording_id"]] = study_mean
        global_baseline.update({r["recording_id"]: mean for r in test})
        by_fold.append({"fold": name, "subjects": len({r["subject_id"] for r in test}),
                        "model": regression_metrics([float(r["regression_label"]) for r in test],
                                                    [prediction[r["recording_id"]] for r in test])})
    by_study = []
    for study in sorted({r["study"] for r in records}):
        test = [r for r in records if r["study"] == study]
        y = [float(r["regression_label"]) for r in test]
        entry = {"study": study, "subjects": len({r["subject_id"] for r in test})}
        for name, values in (("model", prediction), ("training_mean_baseline", global_baseline),
                             ("training_study_mean_baseline", study_baseline)):
            entry[name] = regression_metrics(y, [values[r["recording_id"]] for r in test])
        by_study.append(entry)
    labels = [float(r["regression_label"]) for r in records]
    subject_errors = defaultdict(list)
    for row in records:
        subject_errors[row["subject_id"]].append(abs(prediction[row["recording_id"]]-float(row["regression_label"])))
    report = {"protocol": "mixed_subject", "recordings": len(records), "subjects": len(subject_errors),
              "prediction_level": level, "window_view": window_view, "by_fold": by_fold, "by_study": by_study,
              "equal_subject_mae": sum(sum(v)/len(v) for v in subject_errors.values())/len(subject_errors),
              "notes": ["Every prediction comes from a model that excluded that subject's visits.",
                        "Baseline means use only the corresponding outer training subjects.",
                        "Study-mean baseline is valid only when study identity is known and represented in training.",
                        "Calibration metrics use test labels descriptively; predictions are not recalibrated."]}
    for name, values in (("", prediction), ("_training_mean_baseline", global_baseline),
                         ("_training_study_mean_baseline", study_baseline)):
        source = "model" if not name else name[1:]
        report["pooled"+name] = regression_metrics(labels, [values[r["recording_id"]] for r in records])
        report["macro_study"+name] = {metric: sum(s[source][metric] for s in by_study)/len(by_study)
                                       for metric in ("mae", "rmse")}
    output = [{"recording_id": r["recording_id"], "subject_id": r["subject_id"], "study": r["study"],
               "test_fold": assignment[r["recording_id"]], "truth": float(r["regression_label"]),
               "prediction": prediction[r["recording_id"]],
               "training_mean_baseline": global_baseline[r["recording_id"]],
               "training_study_mean_baseline": study_baseline[r["recording_id"]]} for r in records]
    return report, output
