from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold


POOLING_MODES = ("mean", "mean_temporal_std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether cached WavLM features generalize across studies/sites "
            "and add signal beyond study identity"
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--madrs-threshold", type=float, default=20.0)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--poolings",
        choices=POOLING_MODES,
        nargs="+",
        default=POOLING_MODES,
    )
    parser.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=(100.0, 1000.0, 10000.0)
    )
    parser.add_argument(
        "--c-values", type=float, nargs="+", default=(0.001, 0.01)
    )
    parser.add_argument("--diagnostic-folds", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def infer_study(subject_id: str) -> str:
    value = str(subject_id).strip().upper()
    if value == "THRUIKKS" or value.startswith("MD"):
        return "CDRIN"
    if value.startswith("800"):
        return "FORBOW"
    if value.startswith("VMP"):
        return "VMP"
    if value.startswith("CBN17"):
        return "OPTIMUM-D"
    if value.startswith("TDE"):
        return "TIDE"
    return "UNKNOWN"


def infer_site(subject_id: str) -> str:
    value = str(subject_id).strip().upper()
    study = infer_study(value)
    if study == "OPTIMUM-D":
        match = re.match(r"CBN17[_-]([^_-]+)", value)
        return f"OPTIMUM-D:{match.group(1) if match else 'UNKNOWN'}"
    if study == "TIDE":
        match = re.match(r"TDE\d*[_-]([^_-]+)", value)
        return f"TIDE:{match.group(1) if match else 'UNKNOWN'}"
    return study


def load_recordings(
    input_dir: Path, split: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    layer_means = np.load(input_dir / f"{split}_layer_means.npy", mmap_mode="r")
    layer_stds = np.load(input_dir / f"{split}_layer_stds.npy", mmap_mode="r")
    windows = pd.read_csv(
        input_dir / f"{split}_windows.csv", dtype={"subject_id": str}
    )
    if (
        layer_means.ndim != 3
        or layer_stds.shape != layer_means.shape
        or len(windows) != len(layer_means)
    ):
        raise ValueError(f"Invalid or misaligned {split} cache")
    expected = np.arange(len(windows))
    if not np.array_equal(windows["embedding_index"].to_numpy(), expected):
        raise ValueError(f"Nonsequential {split} embedding indices")

    recording_means: list[np.ndarray] = []
    recording_stds: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for recording_index, group in windows.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(int)
        truth = group["regression_label"].to_numpy(float)
        if not np.allclose(truth, truth[0]):
            raise ValueError(f"Labels differ within recording {recording_index}")
        recording_means.append(
            np.asarray(layer_means[indices], dtype=np.float32).mean(axis=0)
        )
        recording_stds.append(
            np.asarray(layer_stds[indices], dtype=np.float32).mean(axis=0)
        )
        first = group.iloc[0]
        subject_id = str(first["subject_id"]).strip()
        rows.append(
            {
                "recording_index": int(recording_index),
                "audio_path": str(first["audio_path"]),
                "subject_id": subject_id,
                "study": infer_study(subject_id),
                "site": infer_site(subject_id),
                "regression_truth": float(truth[0]),
                "window_count": len(group),
            }
        )
    return np.stack(recording_means), np.stack(recording_stds), pd.DataFrame(rows)


def features_for(
    means: np.ndarray, stds: np.ndarray, layer: int, pooling: str
) -> np.ndarray:
    mean_features = np.asarray(means[:, layer, :], dtype=np.float64)
    if pooling == "mean":
        return mean_features
    if pooling == "mean_temporal_std":
        return np.concatenate(
            [mean_features, np.asarray(stds[:, layer, :], dtype=np.float64)], axis=1
        )
    raise ValueError(pooling)


def standardized(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def balanced_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    weights = {value: len(labels) / (len(values) * count) for value, count in zip(values, counts)}
    return np.asarray([weights[value] for value in labels], dtype=float)


def fit_regression(
    train_features: np.ndarray,
    train_truth: np.ndarray,
    test_features: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaled_train, scaled_test = standardized(train_features, test_features)
    model = Ridge(alpha=alpha)
    model.fit(scaled_train, train_truth)
    return model.predict(scaled_test)


def fit_classification(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    c_value: float,
    seed: int,
    max_iterations: int,
) -> np.ndarray:
    if len(np.unique(train_labels)) != 2:
        raise ValueError("A classification training fold contains only one class")
    scaled_train, scaled_test = standardized(train_features, test_features)
    dual = len(scaled_train) < scaled_train.shape[1]
    model = LogisticRegression(
        C=c_value,
        solver="liblinear",
        dual=dual,
        max_iter=max_iterations,
        random_state=seed,
    )
    model.fit(
        scaled_train,
        train_labels,
        sample_weight=balanced_weights(train_labels),
    )
    return model.predict_proba(scaled_test)[:, 1]


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth_sd = float(np.std(truth, ddof=1))
    prediction_sd = float(np.std(prediction, ddof=1))
    pearson = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if truth_sd > 0 and prediction_sd > 0
        else float("nan")
    )
    return {
        "r2": float(r2_score(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "pearson": pearson,
        "pearson_r2": pearson**2 if np.isfinite(pearson) else float("nan"),
        "prediction_target_sd_ratio": prediction_sd / truth_sd,
        "mean_error": float(np.mean(prediction - truth)),
    }


def safe_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(labels)) != 2:
        return float("nan")
    return float(roc_auc_score(labels, probability))


def choose_cutoff(labels: np.ndarray, probability: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, probability)
    finite = np.isfinite(thresholds)
    score = true_positive[finite] - false_positive[finite]
    candidates = thresholds[finite][np.isclose(score, score.max())]
    return float(candidates[np.argmin(np.abs(candidates - 0.5))])


def classification_metrics(
    labels: np.ndarray, probability: np.ndarray, cutoff: float
) -> dict[str, float]:
    prediction = (probability >= cutoff).astype(int)
    return {
        "roc_auc": safe_auc(labels, probability),
        "average_precision": float(average_precision_score(labels, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "cutoff": float(cutoff),
        "positive_rate": float(np.mean(labels)),
        "predicted_positive_rate": float(np.mean(prediction)),
    }


def domain_oof_regression(
    features: np.ndarray,
    truth: np.ndarray,
    domains: np.ndarray,
    alpha: float,
) -> np.ndarray:
    prediction = np.full(len(truth), np.nan)
    for domain in sorted(np.unique(domains)):
        held_out = domains == domain
        fit = ~held_out
        prediction[held_out] = fit_regression(
            features[fit], truth[fit], features[held_out], alpha
        )
    return prediction


def domain_oof_classification(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    c_value: float,
    seed: int,
    max_iterations: int,
) -> np.ndarray:
    probability = np.full(len(labels), np.nan)
    for index, domain in enumerate(sorted(np.unique(domains))):
        held_out = domains == domain
        fit = ~held_out
        probability[held_out] = fit_classification(
            features[fit],
            labels[fit],
            features[held_out],
            c_value,
            seed + index,
            max_iterations,
        )
    return probability


def one_hot_study(
    train_studies: np.ndarray, test_studies: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    categories = sorted(np.unique(train_studies))
    lookup = {value: index for index, value in enumerate(categories)}
    train = np.zeros((len(train_studies), len(categories)), dtype=float)
    test = np.zeros((len(test_studies), len(categories)), dtype=float)
    for row, value in enumerate(train_studies):
        train[row, lookup[value]] = 1.0
    for row, value in enumerate(test_studies):
        if value in lookup:
            test[row, lookup[value]] = 1.0
    return train, test


def study_baselines(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_labels = (train["regression_truth"].to_numpy(float) >= threshold).astype(int)
    means = train.groupby("study")["regression_truth"].mean()
    rates = pd.Series(train_labels, index=train.index).groupby(train["study"]).mean()
    regression = validation["study"].map(means).fillna(train["regression_truth"].mean())
    probability = validation["study"].map(rates).fillna(train_labels.mean())
    return regression.to_numpy(float), probability.to_numpy(float)


def stratified_group_folds(
    labels: np.ndarray, groups: np.ndarray, count: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=min(count, len(np.unique(groups))), shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(labels)), labels, groups))


def origin_auc(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
    c_value: float,
    folds: int,
    seed: int,
    max_iterations: int,
) -> float:
    features = np.concatenate([train_features, validation_features])
    labels = np.concatenate(
        [np.zeros(len(train_features), dtype=int), np.ones(len(validation_features), dtype=int)]
    )
    groups = np.concatenate([train_groups, validation_groups])
    probability = np.full(len(labels), np.nan)
    for fold_index, (fit, held_out) in enumerate(
        stratified_group_folds(labels, groups, folds, seed), start=1
    ):
        probability[held_out] = fit_classification(
            features[fit],
            labels[fit],
            features[held_out],
            c_value,
            seed + fold_index,
            max_iterations,
        )
    return safe_auc(labels, probability)


def study_predictability(
    features: np.ndarray,
    studies: np.ndarray,
    groups: np.ndarray,
    c_value: float,
    folds: int,
    seed: int,
    max_iterations: int,
) -> float:
    values = sorted(np.unique(studies))
    encoded = np.asarray([values.index(value) for value in studies], dtype=int)
    predictions = np.full(len(encoded), -1)
    for fold_index, (fit, held_out) in enumerate(
        stratified_group_folds(encoded, groups, folds, seed), start=1
    ):
        scaled_fit, scaled_held_out = standardized(features[fit], features[held_out])
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=max_iterations,
            random_state=seed + fold_index,
        )
        model.fit(
            scaled_fit,
            encoded[fit],
            sample_weight=balanced_weights(encoded[fit]),
        )
        predictions[held_out] = model.predict(scaled_held_out)
    return float(balanced_accuracy_score(encoded, predictions))


def cohort_summary(frame: pd.DataFrame, threshold: float, split: str) -> pd.DataFrame:
    data = frame.copy()
    data["binary_truth"] = (data["regression_truth"] >= threshold).astype(int)
    result = (
        data.groupby(["study", "site"], dropna=False)
        .agg(
            recordings=("recording_index", "size"),
            subjects=("subject_id", "nunique"),
            madrs_mean=("regression_truth", "mean"),
            madrs_sd=("regression_truth", "std"),
            positive_n=("binary_truth", "sum"),
            positive_rate=("binary_truth", "mean"),
        )
        .reset_index()
    )
    result.insert(0, "split", split)
    return result


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    best_regression = (
        summary[summary["task"] == "regression"]
        .sort_values("study_loso_rmse")
        .drop_duplicates(["layer_index", "pooling"])
    )
    best_classification = (
        summary[summary["task"] == "classification"]
        .sort_values("study_loso_roc_auc", ascending=False)
        .drop_duplicates(["layer_index", "pooling"])
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for pooling in POOLING_MODES:
        reg = best_regression[best_regression["pooling"] == pooling].sort_values(
            "layer_index"
        )
        cls = best_classification[
            best_classification["pooling"] == pooling
        ].sort_values("layer_index")
        axes[0, 0].plot(reg["layer_index"], reg["study_loso_rmse"], marker="o", label=pooling)
        axes[0, 1].plot(reg["layer_index"], reg["validation_r2"], marker="o", label=pooling)
        axes[1, 0].plot(cls["layer_index"], cls["study_loso_roc_auc"], marker="o", label=pooling)
        axes[1, 1].plot(cls["layer_index"], cls["validation_roc_auc"], marker="o", label=pooling)
    settings = (
        (axes[0, 0], "Leave-one-study-out regression", "RMSE"),
        (axes[0, 1], "External regression (exploratory)", "R²"),
        (axes[1, 0], "Leave-one-study-out classification", "AUROC"),
        (axes[1, 1], "External classification (exploratory)", "AUROC"),
    )
    for axis, title, ylabel in settings:
        axis.set(title=title, xlabel="Layer", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0, 1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1, 0].axhline(0.5, color="black", linestyle="--", alpha=0.5)
    axes[1, 1].axhline(0.5, color="black", linestyle="--", alpha=0.5)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extraction = json.loads((input_dir / "extraction_config.json").read_text())
    layer_names = list(extraction["layer_names"])
    layers = list(range(len(layer_names))) if args.layers is None else list(args.layers)
    if any(layer < 0 or layer >= len(layer_names) for layer in layers):
        raise ValueError(f"Layers must be between 0 and {len(layer_names) - 1}")
    if any(value <= 0 for value in (*args.ridge_alphas, *args.c_values)):
        raise ValueError("Regularization values must be positive")
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": extraction["model_name"],
        "madrs_threshold": args.madrs_threshold,
        "layers": layers,
        "poolings": list(args.poolings),
        "ridge_alphas": list(args.ridge_alphas),
        "c_values": list(args.c_values),
        "diagnostic_folds": args.diagnostic_folds,
        "seed": args.seed,
        "study_mapping": {
            "800*": "FORBOW",
            "MD* and THRUIKKS": "CDRIN",
            "VMP*": "VMP",
            "CBN17*": "OPTIMUM-D",
            "TDE*": "TIDE",
        },
        "selection": "leave-one-study-out training predictions only",
        "external_validation_used_for_selection": False,
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; cached arrays were not loaded.")
        return

    train_means, train_stds, train = load_recordings(input_dir, "train")
    validation_means, validation_stds, validation = load_recordings(
        input_dir, "validation"
    )
    overlap = set(train["subject_id"]) & set(validation["subject_id"])
    if overlap:
        raise ValueError(f"Subject leakage: {sorted(overlap)[:10]}")
    unknown = pd.concat([train, validation]).query("study == 'UNKNOWN'")
    if not unknown.empty:
        raise ValueError(
            "Unknown study IDs: "
            + ", ".join(sorted(unknown["subject_id"].unique())[:20])
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    cohort = pd.concat(
        [
            cohort_summary(train, args.madrs_threshold, "train"),
            cohort_summary(validation, args.madrs_threshold, "validation"),
        ],
        ignore_index=True,
    )
    cohort.to_csv(output_dir / "cohort_summary.csv", index=False)
    train_truth = train["regression_truth"].to_numpy(float)
    validation_truth = validation["regression_truth"].to_numpy(float)
    train_labels = (train_truth >= args.madrs_threshold).astype(int)
    validation_labels = (validation_truth >= args.madrs_threshold).astype(int)
    studies = train["study"].to_numpy(str)
    sites = train["site"].to_numpy(str)

    rows: list[dict[str, object]] = []
    for layer in layers:
        for pooling in args.poolings:
            train_features = features_for(train_means, train_stds, layer, pooling)
            validation_features = features_for(
                validation_means, validation_stds, layer, pooling
            )
            for alpha in args.ridge_alphas:
                oof = domain_oof_regression(
                    train_features, train_truth, studies, alpha
                )
                external = fit_regression(
                    train_features, train_truth, validation_features, alpha
                )
                rows.append(
                    {
                        "task": "regression",
                        "layer_index": layer,
                        "layer_name": layer_names[layer],
                        "pooling": pooling,
                        "regularization": alpha,
                        **{
                            f"study_loso_{key}": value
                            for key, value in regression_metrics(train_truth, oof).items()
                        },
                        **{
                            f"validation_{key}": value
                            for key, value in regression_metrics(
                                validation_truth, external
                            ).items()
                        },
                    }
                )
            for c_value in args.c_values:
                oof = domain_oof_classification(
                    train_features,
                    train_labels,
                    studies,
                    c_value,
                    args.seed + layer,
                    args.max_iterations,
                )
                cutoff = choose_cutoff(train_labels, oof)
                external = fit_classification(
                    train_features,
                    train_labels,
                    validation_features,
                    c_value,
                    args.seed,
                    args.max_iterations,
                )
                rows.append(
                    {
                        "task": "classification",
                        "layer_index": layer,
                        "layer_name": layer_names[layer],
                        "pooling": pooling,
                        "regularization": c_value,
                        **{
                            f"study_loso_{key}": value
                            for key, value in classification_metrics(
                                train_labels, oof, cutoff
                            ).items()
                        },
                        **{
                            f"validation_{key}": value
                            for key, value in classification_metrics(
                                validation_labels, external, cutoff
                            ).items()
                        },
                    }
                )
            print(
                f"layer={layer:02d} pooling={pooling} complete",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "domain_layer_summary.csv", index=False)
    regression_rows = summary[summary["task"] == "regression"]
    classification_rows = summary[summary["task"] == "classification"]
    selected_regression = regression_rows.loc[
        regression_rows["study_loso_rmse"].idxmin()
    ]
    selected_classification = classification_rows.loc[
        classification_rows["study_loso_roc_auc"].idxmax()
    ]

    reg_layer = int(selected_regression["layer_index"])
    reg_pooling = str(selected_regression["pooling"])
    reg_alpha = float(selected_regression["regularization"])
    cls_layer = int(selected_classification["layer_index"])
    cls_pooling = str(selected_classification["pooling"])
    cls_c = float(selected_classification["regularization"])
    reg_train_features = features_for(train_means, train_stds, reg_layer, reg_pooling)
    reg_validation_features = features_for(
        validation_means, validation_stds, reg_layer, reg_pooling
    )
    cls_train_features = features_for(train_means, train_stds, cls_layer, cls_pooling)
    cls_validation_features = features_for(
        validation_means, validation_stds, cls_layer, cls_pooling
    )

    reg_study_oof = domain_oof_regression(
        reg_train_features, train_truth, studies, reg_alpha
    )
    reg_site_oof = domain_oof_regression(
        reg_train_features, train_truth, sites, reg_alpha
    )
    reg_external = fit_regression(
        reg_train_features, train_truth, reg_validation_features, reg_alpha
    )
    cls_study_oof = domain_oof_classification(
        cls_train_features,
        train_labels,
        studies,
        cls_c,
        args.seed,
        args.max_iterations,
    )
    cls_site_oof = domain_oof_classification(
        cls_train_features,
        train_labels,
        sites,
        cls_c,
        args.seed,
        args.max_iterations,
    )
    cutoff = choose_cutoff(train_labels, cls_study_oof)
    cls_external = fit_classification(
        cls_train_features,
        train_labels,
        cls_validation_features,
        cls_c,
        args.seed,
        args.max_iterations,
    )

    study_regression, study_probability = study_baselines(
        train, validation, args.madrs_threshold
    )
    study_cutoff = choose_cutoff(train_labels, train["study"].map(
        pd.Series(train_labels, index=train.index).groupby(train["study"]).mean()
    ).to_numpy(float))

    train_one_hot, validation_one_hot = one_hot_study(
        train["study"].to_numpy(str), validation["study"].to_numpy(str)
    )
    scaled_cls_train, scaled_cls_validation = standardized(
        cls_train_features, cls_validation_features
    )
    combined_probability = fit_classification(
        np.concatenate([scaled_cls_train, train_one_hot], axis=1),
        train_labels,
        np.concatenate([scaled_cls_validation, validation_one_hot], axis=1),
        cls_c,
        args.seed,
        args.max_iterations,
    )
    study_means = train.groupby("study")["regression_truth"].mean()
    train_study_baseline = train["study"].map(study_means).to_numpy(float)
    residual_prediction = fit_regression(
        reg_train_features,
        train_truth - train_study_baseline,
        reg_validation_features,
        reg_alpha,
    )
    combined_regression = study_regression + residual_prediction

    diagnostics = {
        "regression": {
            "selected": {
                "layer_index": reg_layer,
                "layer_name": layer_names[reg_layer],
                "pooling": reg_pooling,
                "alpha": reg_alpha,
            },
            "study_loso_embedding": regression_metrics(train_truth, reg_study_oof),
            "site_loso_embedding": regression_metrics(train_truth, reg_site_oof),
            "external_study_only": regression_metrics(
                validation_truth, study_regression
            ),
            "external_embedding_only": regression_metrics(
                validation_truth, reg_external
            ),
            "external_study_plus_embedding_residual": regression_metrics(
                validation_truth, combined_regression
            ),
        },
        "classification": {
            "selected": {
                "layer_index": cls_layer,
                "layer_name": layer_names[cls_layer],
                "pooling": cls_pooling,
                "c_value": cls_c,
            },
            "study_loso_embedding": classification_metrics(
                train_labels, cls_study_oof, cutoff
            ),
            "site_loso_embedding": classification_metrics(
                train_labels, cls_site_oof, cutoff
            ),
            "external_study_only": classification_metrics(
                validation_labels, study_probability, study_cutoff
            ),
            "external_embedding_only": classification_metrics(
                validation_labels, cls_external, cutoff
            ),
            "external_study_plus_embedding": classification_metrics(
                validation_labels, combined_probability, cutoff
            ),
        },
        "domain_diagnostics": {
            "train_vs_validation_origin_auc": origin_auc(
                cls_train_features,
                cls_validation_features,
                train["subject_id"].to_numpy(str),
                validation["subject_id"].to_numpy(str),
                cls_c,
                args.diagnostic_folds,
                args.seed,
                args.max_iterations,
            ),
            "training_study_balanced_accuracy": study_predictability(
                cls_train_features,
                studies,
                train["subject_id"].to_numpy(str),
                cls_c,
                args.diagnostic_folds,
                args.seed,
                args.max_iterations,
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )

    predictions = validation.copy()
    predictions["binary_truth"] = validation_labels
    predictions["study_regression_prediction"] = study_regression
    predictions["embedding_regression_prediction"] = reg_external
    predictions["combined_regression_prediction"] = combined_regression
    predictions["study_probability"] = study_probability
    predictions["embedding_probability"] = cls_external
    predictions["combined_probability"] = combined_probability
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)

    within_rows = []
    for study, indices in validation.groupby("study").groups.items():
        index = np.asarray(list(indices), dtype=int)
        for model_name, regression_prediction, probability in (
            ("study_only", study_regression, study_probability),
            ("embedding_only", reg_external, cls_external),
            ("study_plus_embedding", combined_regression, combined_probability),
        ):
            reg = regression_metrics(validation_truth[index], regression_prediction[index])
            within_rows.append(
                {
                    "study": study,
                    "model": model_name,
                    "recordings": len(index),
                    "positive_n": int(validation_labels[index].sum()),
                    **reg,
                    "roc_auc": safe_auc(validation_labels[index], probability[index]),
                }
            )
    pd.DataFrame(within_rows).to_csv(
        output_dir / "within_study_validation_metrics.csv", index=False
    )
    plot_summary(summary, output_dir / "domain_layer_curves.png")

    reg = diagnostics["regression"]
    cls = diagnostics["classification"]
    domain = diagnostics["domain_diagnostics"]
    lines = [
        f"# WavLM domain-generalization audit: {extraction['model_name']}",
        "",
        "Representations and regularization were selected using leave-one-study-out",
        "training predictions only. External results did not control selection.",
        "`THRUIKKS` is mapped to CDRIN.",
        "",
        "## Selected representations",
        "",
        f"- Regression: layer {reg_layer} (`{layer_names[reg_layer]}`), "
        f"`{reg_pooling}`, alpha {reg_alpha:g}.",
        f"- Classification: layer {cls_layer} (`{layer_names[cls_layer]}`), "
        f"`{cls_pooling}`, C {cls_c:g}.",
        "",
        "## Regression",
        "",
        "| Evaluation | R² | RMSE | Pearson | Pearson² | Pred/target SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("Leave-one-study-out embedding", "study_loso_embedding"),
        ("Leave-one-site-out embedding", "site_loso_embedding"),
        ("External study only", "external_study_only"),
        ("External embedding only", "external_embedding_only"),
        ("External study + embedding residual", "external_study_plus_embedding_residual"),
    ):
        value = reg[key]
        lines.append(
            f"| {label} | {value['r2']:.3f} | {value['rmse']:.3f} | "
            f"{value['pearson']:.3f} | {value['pearson_r2']:.3f} | "
            f"{value['prediction_target_sd_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## MADRS classification",
            "",
            "| Evaluation | AUROC | AP | Balanced accuracy | Macro F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("Leave-one-study-out embedding", "study_loso_embedding"),
        ("Leave-one-site-out embedding", "site_loso_embedding"),
        ("External study only", "external_study_only"),
        ("External embedding only", "external_embedding_only"),
        ("External study + embedding", "external_study_plus_embedding"),
    ):
        value = cls[key]
        lines.append(
            f"| {label} | {value['roc_auc']:.3f} | {value['average_precision']:.3f} | "
            f"{value['balanced_accuracy']:.3f} | {value['macro_f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Domain predictability",
            "",
            f"- Train-versus-validation origin AUROC: "
            f"{domain['train_vs_validation_origin_auc']:.3f}.",
            f"- Training-study balanced accuracy from embeddings: "
            f"{domain['training_study_balanced_accuracy']:.3f} "
            "(chance is 0.200 for five studies).",
            "",
            "See `cohort_summary.csv`, `domain_layer_summary.csv`,",
            "`within_study_validation_metrics.csv`, and `validation_predictions.csv`",
            "for the detailed audit. External per-layer values are exploratory.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
