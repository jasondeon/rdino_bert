from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation_metrics import intraclass_correlation_2_1


POOLING_MODES = ("mean", "mean_between_window_std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Elastic Net and random-forest baselines to IS09 features"
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--madrs-threshold", type=float, default=20.0)
    parser.add_argument("--rf-search-iterations", type=int, default=24)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=-1)
    return parser.parse_args()


def load_split(
    input_dir: Path, split: str
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    features = np.load(input_dir / f"{split}_features.npy", mmap_mode="r")
    windows = pd.read_csv(
        input_dir / f"{split}_windows.csv", dtype={"subject_id": str}
    )
    if features.ndim != 2 or features.shape[1] != 384 or len(features) != len(windows):
        raise ValueError(f"Invalid or misaligned {split} IS09 cache")
    if not np.array_equal(windows["embedding_index"], np.arange(len(windows))):
        raise ValueError(f"Nonsequential {split} embedding indices")
    mean_rows = []
    std_rows = []
    metadata = []
    for recording_index, group in windows.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(int)
        values = np.asarray(features[indices], dtype=np.float64)
        scores = group["regression_label"].to_numpy(float)
        if not np.allclose(scores, scores[0]):
            raise ValueError(f"Labels differ within recording {recording_index}")
        mean_rows.append(values.mean(axis=0))
        std_rows.append(values.std(axis=0, ddof=0))
        first = group.iloc[0]
        metadata.append(
            {
                "recording_index": int(recording_index),
                "audio_path": str(first["audio_path"]),
                "subject_id": str(first["subject_id"]),
                "regression_truth": float(scores[0]),
                "window_count": len(indices),
            }
        )
    return np.stack(mean_rows), np.stack(std_rows), pd.DataFrame(metadata)


def pooled_features(means: np.ndarray, stds: np.ndarray, pooling: str) -> np.ndarray:
    if pooling in {"single_crop", "mean"}:
        return means
    if pooling == "mean_between_window_std":
        return np.concatenate((means, stds), axis=1)
    raise ValueError(pooling)


def feature_names(base_names: list[str], pooling: str) -> list[str]:
    if pooling == "single_crop":
        return base_names
    names = [f"mean__{name}" for name in base_names]
    if pooling == "mean_between_window_std":
        names.extend(f"between_window_std__{name}" for name in base_names)
    return names


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "icc_2_1": intraclass_correlation_2_1(truth, prediction),
        "pearson": float(np.corrcoef(truth, prediction)[0, 1]),
        "mean_error": float(np.mean(prediction - truth)),
        "prediction_target_sd_ratio": float(
            np.std(prediction, ddof=1) / np.std(truth, ddof=1)
        ),
    }


def choose_cutoff(truth: np.ndarray, probability: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(truth, probability)
    finite = np.isfinite(thresholds)
    scores = true_positive[finite] - false_positive[finite]
    candidates = thresholds[finite][np.isclose(scores, scores.max())]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def classification_metrics(
    truth: np.ndarray, probability: np.ndarray, cutoff: float
) -> dict[str, float]:
    prediction = (probability >= cutoff).astype(int)
    return {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "average_precision": float(average_precision_score(truth, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "mcc": float(matthews_corrcoef(truth, prediction)),
        "sensitivity": float(recall_score(truth, prediction)),
        "specificity": float(np.mean(prediction[truth == 0] == 0)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "cutoff": cutoff,
    }


def subject_bootstrap(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    metric: str,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    groups = {
        subject: np.asarray(indices, dtype=int)
        for subject, indices in frame.groupby("subject_id").groups.items()
    }
    subjects = np.asarray(list(groups), dtype=object)
    truth = frame[truth_column].to_numpy()
    prediction = frame[prediction_column].to_numpy(float)
    function = r2_score if metric == "r2" else roc_auc_score
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected = rng.choice(subjects, len(subjects), replace=True)
        indices = np.concatenate([groups[subject] for subject in selected])
        if len(np.unique(truth[indices])) > 1:
            values.append(function(truth[indices], prediction[indices]))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def estimator_and_search(
    model_name: str,
    outcome: str,
    args: argparse.Namespace,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> GridSearchCV | RandomizedSearchCV:
    if model_name == "elastic_net" and outcome == "regression":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", ElasticNet(max_iter=50_000, random_state=args.seed)),
            ]
        )
        return GridSearchCV(
            estimator,
            {
                "model__alpha": np.logspace(-4, 3, 15),
                "model__l1_ratio": (0.1, 0.5, 0.9, 1.0),
            },
            scoring="neg_root_mean_squared_error",
            cv=folds,
            n_jobs=args.jobs,
            refit=True,
        )
    if model_name == "elastic_net" and outcome == "classification":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        class_weight="balanced",
                        max_iter=20_000,
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        return GridSearchCV(
            estimator,
            {
                "model__C": np.logspace(-3, 3, 13),
                "model__l1_ratio": (0.1, 0.5, 0.9, 1.0),
            },
            scoring="roc_auc",
            cv=folds,
            n_jobs=args.jobs,
            refit=True,
        )
    common_parameters = {
        "max_depth": (None, 3, 5, 8, 12, 20),
        "min_samples_leaf": (1, 2, 4, 8, 16),
        "min_samples_split": (2, 5, 10, 20),
        "max_features": ("sqrt", 0.25, 0.5, 1.0),
    }
    if outcome == "regression":
        estimator = RandomForestRegressor(
            n_estimators=args.rf_trees,
            criterion="squared_error",
            random_state=args.seed,
            n_jobs=1,
        )
        scoring = "neg_root_mean_squared_error"
    else:
        estimator = RandomForestClassifier(
            n_estimators=args.rf_trees,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=1,
        )
        scoring = "roc_auc"
    return RandomizedSearchCV(
        estimator,
        common_parameters,
        n_iter=args.rf_search_iterations,
        scoring=scoring,
        cv=folds,
        random_state=args.seed,
        n_jobs=args.jobs,
        refit=True,
    )


def importance_table(
    estimator: Any, names: list[str], model_name: str, outcome: str
) -> pd.DataFrame:
    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    if model_name == "elastic_net":
        values = np.asarray(model.coef_).reshape(-1)
    else:
        values = np.asarray(model.feature_importances_)
    return pd.DataFrame(
        {
            "feature": names,
            "importance": values,
            "absolute_importance": np.abs(values),
            "model": model_name,
            "outcome": outcome,
        }
    ).sort_values("absolute_importance", ascending=False)


def main() -> None:
    args = parse_args()
    if args.cv_folds < 2 or args.bootstrap_samples < 100:
        raise ValueError("Use at least two folds and 100 bootstrap samples")
    input_dir = args.input_dir.expanduser().resolve()
    extraction = json.loads((input_dir / "extraction_config.json").read_text())
    if extraction["feature_set"] != "IS09" or extraction["feature_count"] != 384:
        raise ValueError("Input directory is not an IS09 functional cache")
    base_names = pd.read_csv(input_dir / "feature_names.csv")["feature_name"].tolist()
    train_means, train_stds, train = load_split(input_dir, "train")
    validation_means, validation_stds, validation = load_split(input_dir, "validation")
    if set(train["subject_id"]) & set(validation["subject_id"]):
        raise ValueError("Training and validation data share subjects")

    window_selection = extraction.get("window_selection", "all")
    pooling_modes = (
        ("single_crop",)
        if window_selection == "first_per_recording"
        else POOLING_MODES
    )
    if window_selection == "first_per_recording" and not (
        train["window_count"].eq(1).all() and validation["window_count"].eq(1).all()
    ):
        raise ValueError("first_per_recording extraction contains multiple crops")
    regression_truth = train["regression_truth"].to_numpy(float)
    validation_regression_truth = validation["regression_truth"].to_numpy(float)
    binary_truth = (regression_truth >= args.madrs_threshold).astype(int)
    validation_binary_truth = (
        validation_regression_truth >= args.madrs_threshold
    ).astype(int)
    groups = train["subject_id"].astype(str).to_numpy()
    regression_folds = list(GroupKFold(args.cv_folds).split(regression_truth, groups=groups))
    classification_folds = list(
        StratifiedGroupKFold(
            args.cv_folds, shuffle=True, random_state=args.seed
        ).split(regression_truth, binary_truth, groups)
    )
    config = {
        "input_dir": str(input_dir),
        "feature_set": "IS09 Functionals",
        "window_seconds": extraction["window_seconds"],
        "stride_seconds": extraction["stride_seconds"],
        "window_selection": window_selection,
        "pooling_modes": pooling_modes,
        "models": ("elastic_net", "random_forest"),
        "outcomes": ("regression", f"MADRS >= {args.madrs_threshold:g}"),
        "cv_folds": args.cv_folds,
        "rf_search_iterations": args.rf_search_iterations,
        "rf_trees": args.rf_trees,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "external_validation_used_for_selection": False,
    }
    (input_dir / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")
    validation_predictions = validation.copy()
    validation_predictions["binary_truth"] = validation_binary_truth
    search_rows = []
    importance_rows = []
    results: dict[str, Any] = {"configuration": config, "models": {}}
    training_mean = float(regression_truth.mean())
    training_mean_prediction = np.full_like(
        validation_regression_truth, training_mean, dtype=float
    )
    validation_predictions["prediction_training_mean"] = training_mean_prediction
    results["training_mean_baseline"] = {
        "training_mean": training_mean,
        "validation": {
            "r2": float(r2_score(validation_regression_truth, training_mean_prediction)),
            "rmse": float(
                np.sqrt(mean_squared_error(validation_regression_truth, training_mean_prediction))
            ),
            "mean_error": float(np.mean(training_mean_prediction - validation_regression_truth)),
        },
    }

    for outcome in ("regression", "classification"):
        truth = regression_truth if outcome == "regression" else binary_truth
        validation_truth = (
            validation_regression_truth if outcome == "regression" else validation_binary_truth
        )
        folds = regression_folds if outcome == "regression" else classification_folds
        for model_name in ("elastic_net", "random_forest"):
            candidates = []
            for pooling in pooling_modes:
                train_features = pooled_features(train_means, train_stds, pooling)
                search = estimator_and_search(model_name, outcome, args, folds)
                search.fit(train_features, truth)
                score = float(search.best_score_)
                candidates.append((score, pooling, search))
                search_rows.append(
                    {
                        "outcome": outcome,
                        "model": model_name,
                        "pooling": pooling,
                        "cv_score": score,
                        "cv_metric": "neg_rmse" if outcome == "regression" else "roc_auc",
                        "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                    }
                )
                print(
                    f"outcome={outcome} model={model_name} pooling={pooling} "
                    f"cv_score={score:.4f}",
                    flush=True,
                )
            _, selected_pooling, selected_search = max(candidates, key=lambda item: item[0])
            train_features = pooled_features(train_means, train_stds, selected_pooling)
            validation_features = pooled_features(
                validation_means, validation_stds, selected_pooling
            )
            estimator = selected_search.best_estimator_
            method = "predict" if outcome == "regression" else "predict_proba"
            oof = cross_val_predict(
                clone(estimator),
                train_features,
                truth,
                cv=folds,
                n_jobs=args.jobs,
                method=method,
            )
            if outcome == "classification":
                oof = oof[:, 1]
            estimator.fit(train_features, truth)
            prediction = (
                estimator.predict(validation_features)
                if outcome == "regression"
                else estimator.predict_proba(validation_features)[:, 1]
            )
            key = f"{outcome}_{model_name}"
            if outcome == "regression":
                oof_metrics = regression_metrics(truth, oof)
                validation_metrics = regression_metrics(validation_truth, prediction)
                validation_predictions[f"prediction_{key}"] = prediction
                ci = subject_bootstrap(
                    validation_predictions,
                    "regression_truth",
                    f"prediction_{key}",
                    "r2",
                    samples=args.bootstrap_samples,
                    seed=args.seed + len(results["models"]),
                )
                validation_metrics["r2_ci"] = ci
            else:
                cutoff = choose_cutoff(truth, oof)
                oof_metrics = classification_metrics(truth, oof, cutoff)
                validation_metrics = classification_metrics(
                    validation_truth, prediction, cutoff
                )
                validation_predictions[f"probability_{key}"] = prediction
                ci = subject_bootstrap(
                    validation_predictions,
                    "binary_truth",
                    f"probability_{key}",
                    "auc",
                    samples=args.bootstrap_samples,
                    seed=args.seed + 100 + len(results["models"]),
                )
                validation_metrics["roc_auc_ci"] = ci
            results["models"][key] = {
                "pooling": selected_pooling,
                "feature_dimension": int(train_features.shape[1]),
                "parameters": selected_search.best_params_,
                "selection_cv_score": float(selected_search.best_score_),
                "oof": oof_metrics,
                "validation": validation_metrics,
            }
            joblib.dump(estimator, input_dir / f"model_{key}.joblib")
            importance_rows.append(
                importance_table(
                    estimator,
                    feature_names(base_names, selected_pooling),
                    model_name,
                    outcome,
                ).head(100)
            )
            print(
                f"SELECTED outcome={outcome} model={model_name} pooling={selected_pooling}",
                flush=True,
            )

    pd.DataFrame(search_rows).to_csv(input_dir / "model_search_summary.csv", index=False)
    pd.concat(importance_rows, ignore_index=True).to_csv(
        input_dir / "feature_importances.csv", index=False
    )
    validation_predictions.to_csv(input_dir / "validation_predictions.csv", index=False)
    (input_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    if window_selection == "first_per_recording":
        extraction_description = [
            "Official IS09 functionals were extracted once from the earliest valid",
            "continuous primary-speaker crop for each recording. Natural unlabeled gaps",
            "inside the crop were preserved, and crops do not cross a detected speaker switch.",
        ]
    else:
        extraction_description = [
            "Official IS09 functionals were extracted independently from each diarization-aware",
            "audio window and aggregated by recording.",
        ]
    lines = [
        "# openSMILE IS09 acoustic baselines",
        "",
        *extraction_description,
        "Pooling and hyperparameters were selected",
        "using subject-grouped training CV only. Confidence intervals resample validation",
        "subjects as clusters.",
        "",

        "## MADRS regression",
        "",
        "| Model | Pooling | OOF RMSE | Validation RMSE | Validation R² [95% CI] | ICC | Pearson |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    baseline = results["training_mean_baseline"]
    lines.append(
        "| training mean | — | — | "
        f"{baseline['validation']['rmse']:.3f} | "
        f"{baseline['validation']['r2']:.3f} | — | — |"
    )


    for model_name in ("elastic_net", "random_forest"):
        value = results["models"][f"regression_{model_name}"]
        ci = value["validation"]["r2_ci"]
        lines.append(
            f"| {model_name} | {value['pooling']} | {value['oof']['rmse']:.3f} | "
            f"{value['validation']['rmse']:.3f} | {value['validation']['r2']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {value['validation']['icc_2_1']:.3f} | "
            f"{value['validation']['pearson']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"## MADRS ≥{args.madrs_threshold:g} classification",
            "",
            "| Model | Pooling | OOF AUROC | Validation AUROC [95% CI] | AP | Balanced accuracy | Macro F1 | MCC |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name in ("elastic_net", "random_forest"):
        value = results["models"][f"classification_{model_name}"]
        ci = value["validation"]["roc_auc_ci"]
        lines.append(
            f"| {model_name} | {value['pooling']} | {value['oof']['roc_auc']:.3f} | "
            f"{value['validation']['roc_auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{value['validation']['average_precision']:.3f} | "
            f"{value['validation']['balanced_accuracy']:.3f} | "
            f"{value['validation']['macro_f1']:.3f} | {value['validation']['mcc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "OOF metrics reuse the CV process that selected hyperparameters and pooling and",
            "are therefore model-selection diagnostics rather than unbiased performance",
            "estimates. External validation was not used for selection.",
        ]
    )
    (input_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {input_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
