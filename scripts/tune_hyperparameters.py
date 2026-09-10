from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.trial import TrialState


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingFailure(RuntimeError):
    """A training subprocess did not produce a usable history."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable Optuna search, then confirm the strongest "
            "RDINO-BERT configurations across additional seeds"
        )
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("/data/Clinical_vars/canbind_combined_20260806_train.csv"),
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=Path(
            "/data/Clinical_vars/canbind_combined_20260806_validation.csv"
        ),
    )
    parser.add_argument(
        "--rdino-checkpoint", type=Path, default=Path("assets/pretrained_rdino.pth")
    )
    parser.add_argument("--study-name", default="rdino-bert-existing-settings-v1")
    parser.add_argument(
        "--study-dir", type=Path, default=Path("experiments/hpo_existing_settings")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/hpo"))
    parser.add_argument(
        "--n-trials",
        type=int,
        default=12,
        help="Target total screening trials in the resumable study (default: 12)",
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--confirmation-seeds",
        type=int,
        nargs="*",
        default=[41, 42],
        help="Extra seeds for each top configuration (screening uses seed 40)",
    )
    parser.add_argument("--screening-seed", type=int, default=40)
    parser.add_argument("--sampler-seed", type=int, default=2026)
    parser.add_argument(
        "--pruning-startup-trials",
        type=int,
        default=5,
        help="Complete this many trials before median pruning is enabled",
    )
    parser.add_argument(
        "--pruning-warmup-epochs",
        type=int,
        default=3,
        help="Never prune a trial before this many validation epochs",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--stride-seconds", type=float, default=20.0)
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Run only the Optuna screening stage",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and describe the search without launching training",
    )
    return parser.parse_args()


def resolved_from_project(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1")
    if args.pruning_startup_trials < 1:
        raise ValueError("--pruning-startup-trials must be at least 1")
    if args.pruning_warmup_epochs < 1:
        raise ValueError("--pruning-warmup-epochs must be at least 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 because the model uses BatchNorm")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.window_seconds <= 0 or args.stride_seconds <= 0:
        raise ValueError("Window and stride must be positive")
    if len(set(args.confirmation_seeds)) != len(args.confirmation_seeds):
        raise ValueError("--confirmation-seeds cannot contain duplicates")
    if args.screening_seed in args.confirmation_seeds:
        raise ValueError("Confirmation seeds must differ from --screening-seed")

    for label, path in (
        ("training manifest", resolved_from_project(args.train_manifest)),
        ("validation manifest", resolved_from_project(args.validation_manifest)),
        ("RDINO checkpoint", resolved_from_project(args.rdino_checkpoint)),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")


def common_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "train_manifest": str(resolved_from_project(args.train_manifest)),
        "validation_manifest": str(resolved_from_project(args.validation_manifest)),
        "num_classes": 4,
        "rdino_checkpoint": str(resolved_from_project(args.rdino_checkpoint)),
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_sampling": "window",
        "regression_weight": 1.0,
        "class_weighting": "sqrt_inverse_frequency",
        "workers": args.workers,
        "early_stopping_patience": args.early_stopping_patience,
        "standardize_regression_labels": True,
        "lr_scheduler": "plateau",
        "min_learning_rate": 1e-7,
    }


def parameter_arguments(parameters: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for name, value in parameters.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            result.extend((flag, str(value)))
    return result


def training_command(parameters: dict[str, Any], output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "train.py"),
        *parameter_arguments(parameters),
        "--output-dir",
        str(output_dir),
    ]


def run_training(
    run_id: str,
    parameters: dict[str, Any],
    output_dir: Path,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        history_path = output_dir / "training_history.csv"
        if history_path.is_file():
            print(f"Reusing completed run {run_id}: {output_dir}")
            return best_history_row(history_path)
        raise FileExistsError(
            f"Refusing to overwrite incomplete output directory: {output_dir}"
        )

    output_dir.mkdir(parents=True)
    command = training_command(parameters, output_dir)
    metadata_path = output_dir / "hpo_run.json"
    metadata = {
        "run_id": run_id,
        "status": "running",
        "parameters": parameters,
        "command": command,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== {run_id} ===")
    print(shlex.join(command), flush=True)

    process: subprocess.Popen[str] | None = None
    started = time.monotonic()
    exit_code = -1
    reported_epochs = 0
    pruned = False
    try:
        with (output_dir / "training.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                history_path = output_dir / "training_history.csv"
                if trial is not None and history_path.is_file():
                    try:
                        history = pd.read_csv(history_path)
                    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                        history = pd.DataFrame()
                    if len(history) > reported_epochs:
                        new_rows = history.iloc[reported_epochs:]
                        for _, history_row in new_rows.iterrows():
                            epoch = int(history_row["epoch"])
                            validation_r2 = float(
                                history_row["validation_regression_r2"]
                            )
                            trial.report(validation_r2, step=epoch)
                            if trial.should_prune():
                                print(
                                    f"Pruning {run_id} after epoch {epoch} "
                                    f"(recording R²={validation_r2:.6f})",
                                    flush=True,
                                )
                                pruned = True
                                process.terminate()
                                break
                        reported_epochs = len(history)
                    if pruned:
                        break
            exit_code = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        exit_code = 130
        raise
    finally:
        metadata["duration_seconds"] = round(time.monotonic() - started, 1)
        metadata["exit_code"] = exit_code

    if pruned:
        metadata["status"] = "pruned"
        metadata["reported_epochs"] = reported_epochs
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        raise optuna.TrialPruned(f"Median pruner stopped {run_id}")

    history_path = output_dir / "training_history.csv"
    if exit_code != 0 or not history_path.is_file():
        metadata["status"] = "failed"
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        raise TrainingFailure(f"{run_id} exited with code {exit_code}")

    best = best_history_row(history_path)
    metadata.update(
        {
            "status": "completed",
            "best_epoch": int(best["epoch"]),
            "best_validation_regression_r2": float(
                best["validation_regression_r2"]
            ),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "plot_training.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return best


def best_history_row(path: Path) -> dict[str, Any]:
    history = pd.read_csv(path)
    if history.empty:
        raise TrainingFailure(f"Training history is empty: {path}")
    row = history.loc[history["validation_regression_r2"].idxmax()]
    return row.to_dict()


def suggested_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 3e-6, 3e-5, log=True
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", [0.0, 1e-6, 1e-5, 1e-4]
        ),
        "classification_weight": trial.suggest_float(
            "classification_weight", 0.75, 4.0, log=True
        ),
        "lr_scheduler_factor": trial.suggest_categorical(
            "lr_scheduler_factor", [0.3, 0.5, 0.7]
        ),
        "lr_scheduler_patience": trial.suggest_int(
            "lr_scheduler_patience", 1, 3
        ),
    }


def record_trial_metrics(
    trial: optuna.Trial, best: dict[str, Any], output_dir: Path
) -> None:
    metric_columns = (
        "validation_regression_rmse",
        "validation_regression_rmse_original_scale",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "validation_accuracy",
        "validation_classification_loss",
        "validation_regression_loss",
    )
    trial.set_user_attr("best_epoch", int(best["epoch"]))
    trial.set_user_attr("output_dir", str(output_dir.relative_to(PROJECT_ROOT)))
    for column in metric_columns:
        trial.set_user_attr(column, float(best[column]))


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return sorted(
        (
            trial
            for trial in study.trials
            if trial.state == TrialState.COMPLETE and trial.value is not None
        ),
        key=lambda trial: float(trial.value),
        reverse=True,
    )


def save_screening_artifacts(study: optuna.Study, study_dir: Path) -> None:
    study.trials_dataframe().to_csv(study_dir / "screening_trials.csv", index=False)
    completed = completed_trials(study)
    importances: dict[str, float] = {}
    if len(completed) >= 3:
        try:
            importances = get_param_importances(study)
        except (RuntimeError, ValueError):
            pass
    (study_dir / "parameter_importance.json").write_text(
        json.dumps(importances, indent=2) + "\n", encoding="utf-8"
    )


def confirmation_row(
    trial: optuna.trial.FrozenTrial,
    seed: int,
    source: str,
    best: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    return {
        "source_trial": trial.number,
        "source": source,
        "seed": seed,
        **trial.params,
        "best_epoch": int(best["epoch"]),
        "validation_regression_r2": float(best["validation_regression_r2"]),
        "validation_regression_rmse": float(best["validation_regression_rmse"]),
        "validation_regression_rmse_original_scale": float(
            best["validation_regression_rmse_original_scale"]
        ),
        "validation_balanced_accuracy": float(
            best["validation_balanced_accuracy"]
        ),
        "validation_macro_f1": float(best["validation_macro_f1"]),
        "output_dir": output_dir,
    }


def confirmation_runs(
    study: optuna.Study,
    args: argparse.Namespace,
    shared: dict[str, Any],
    study_dir: Path,
    output_root: Path,
) -> pd.DataFrame:
    top_trials = completed_trials(study)[: args.top_k]
    rows: list[dict[str, Any]] = []
    for trial in top_trials:
        screening_history = Path(trial.user_attrs["output_dir"]) / "training_history.csv"
        screening_history = resolved_from_project(screening_history)
        rows.append(
            confirmation_row(
                trial,
                args.screening_seed,
                "screening",
                best_history_row(screening_history),
                trial.user_attrs["output_dir"],
            )
        )
        for seed in args.confirmation_seeds:
            run_id = f"trial-{trial.number:04d}-seed-{seed}"
            output_dir = output_root / "confirmation" / run_id
            parameters = {**shared, **trial.params, "seed": seed}
            best = run_training(run_id, parameters, output_dir)
            rows.append(
                confirmation_row(
                    trial,
                    seed,
                    "confirmation",
                    best,
                    str(output_dir.relative_to(PROJECT_ROOT)),
                )
            )
            pd.DataFrame(rows).to_csv(
                study_dir / "confirmation_runs.csv", index=False
            )
    return pd.DataFrame(rows)


def save_confirmation_summary(frame: pd.DataFrame, study_dir: Path) -> pd.DataFrame:
    if frame.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            frame.groupby("source_trial", as_index=False)
            .agg(
                seeds=("seed", "count"),
                mean_validation_regression_r2=("validation_regression_r2", "mean"),
                std_validation_regression_r2=("validation_regression_r2", "std"),
                mean_validation_regression_rmse=(
                    "validation_regression_rmse",
                    "mean",
                ),
                mean_validation_balanced_accuracy=(
                    "validation_balanced_accuracy",
                    "mean",
                ),
                mean_validation_macro_f1=("validation_macro_f1", "mean"),
            )
            .sort_values("mean_validation_regression_r2", ascending=False)
        )
    summary.to_csv(study_dir / "confirmation_summary.csv", index=False)
    return summary


def write_report(
    study: optuna.Study,
    study_dir: Path,
    confirmation_summary: pd.DataFrame | None,
) -> None:
    ranked = completed_trials(study)
    lines = [
        "# RDINO-BERT hyperparameter search",
        "",
        "The screening objective is best-epoch, full-recording validation R².",
        "Window size/stride, sampling, class weighting, augmentation, and stride",
        "jitter were held fixed based on the earlier experiments.",
        "",
        "## Best screening trials",
        "",
        "| Trial | R² | RMSE | Balanced accuracy | Macro F1 | Best epoch | Parameters |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trial in ranked[:10]:
        lines.append(
            f"| {trial.number} | {float(trial.value):.6f} | "
            f"{trial.user_attrs['validation_regression_rmse']:.6f} | "
            f"{trial.user_attrs['validation_balanced_accuracy']:.6f} | "
            f"{trial.user_attrs['validation_macro_f1']:.6f} | "
            f"{trial.user_attrs['best_epoch']} | `{json.dumps(trial.params, sort_keys=True)}` |"
        )
    if not ranked:
        lines.append("| - | - | - | - | - | - | No completed trials |")

    importance_path = study_dir / "parameter_importance.json"
    importances = (
        json.loads(importance_path.read_text(encoding="utf-8"))
        if importance_path.is_file()
        else {}
    )
    lines.extend(["", "## Parameter importance", ""])
    if importances:
        for parameter, importance in importances.items():
            lines.append(f"- `{parameter}`: {importance:.4f}")
    else:
        lines.append("Not enough completed trials to estimate reliably.")

    lines.extend(["", "## Cross-seed confirmation", ""])
    if confirmation_summary is not None and not confirmation_summary.empty:
        lines.extend(
            [
                "| Source trial | Seeds | Mean R² | R² SD | Mean RMSE | Mean balanced accuracy | Mean macro F1 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in confirmation_summary.iterrows():
            lines.append(
                f"| {int(row['source_trial'])} | {int(row['seeds'])} | "
                f"{row['mean_validation_regression_r2']:.6f} | "
                f"{row['std_validation_regression_r2']:.6f} | "
                f"{row['mean_validation_regression_rmse']:.6f} | "
                f"{row['mean_validation_balanced_accuracy']:.6f} | "
                f"{row['mean_validation_macro_f1']:.6f} |"
            )
    else:
        lines.append("Confirmation has not been run.")

    lines.extend(
        [
            "",
            "## Interpretation checklist",
            "",
            "- Prefer cross-seed mean R² over the single best screening run.",
            "- Treat differences below 0.02 R² as inconclusive at this sample size.",
            "- If all settings converge to similar R², investigate data, modality,",
            "  aggregation, gradient-flow, or representation failure modes next.",
            "- Use the saved histories and recording predictions to inspect overfitting,",
            "  residuals, class confusion, recording length, and site effects.",
        ]
    )
    (study_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe_dry_run(args: argparse.Namespace, shared: dict[str, Any]) -> None:
    maximum_runs = args.n_trials
    if not args.skip_confirmation:
        maximum_runs += args.top_k * len(args.confirmation_seeds)
    print("Validated standalone HPO configuration.")
    print(f"Screening trials: {args.n_trials}")
    print(f"Top configurations confirmed: {0 if args.skip_confirmation else args.top_k}")
    print(f"Additional confirmation seeds: {args.confirmation_seeds}")
    print(f"Maximum total GPU runs: {maximum_runs}")
    print(
        "Median pruning: enabled after "
        f"{args.pruning_startup_trials} startup trials and "
        f"{args.pruning_warmup_epochs} warmup epochs"
    )
    print("Fixed settings:")
    print(json.dumps(shared, indent=2, sort_keys=True))
    print("Search space:")
    print(
        json.dumps(
            {
                "learning_rate": "log-uniform [3e-6, 3e-5]",
                "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
                "classification_weight": "log-uniform [0.75, 4.0]",
                "lr_scheduler_factor": [0.3, 0.5, 0.7],
                "lr_scheduler_patience": "integer [1, 3]",
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    shared = common_parameters(args)
    study_dir = resolved_from_project(args.study_dir)
    output_root = resolved_from_project(args.output_root)
    if not study_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("--study-dir must be inside the project")
    if not output_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("--output-root must be inside the project")

    if args.dry_run:
        describe_dry_run(args, shared)
        return

    study_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{study_dir / 'study.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.sampler_seed,
            n_startup_trials=args.pruning_startup_trials,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.pruning_startup_trials,
            n_warmup_steps=args.pruning_warmup_epochs,
            interval_steps=1,
        ),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = {
            **shared,
            **suggested_parameters(trial),
            "seed": args.screening_seed,
        }
        output_dir = output_root / "screening" / f"trial-{trial.number:04d}"
        best = run_training(
            f"screening-trial-{trial.number:04d}",
            parameters,
            output_dir,
            trial=trial,
        )
        record_trial_metrics(trial, best, output_dir)
        return float(best["validation_regression_r2"])

    remaining_trials = max(0, args.n_trials - len(study.trials))
    if remaining_trials:
        print(
            f"Running {remaining_trials} screening trials to reach target total "
            f"{args.n_trials}."
        )
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[lambda current, _: save_screening_artifacts(current, study_dir)],
            gc_after_trial=True,
            n_jobs=1,
        )
    else:
        print(f"Study already has {len(study.trials)} trials; screening is complete.")
    save_screening_artifacts(study, study_dir)

    confirmation_summary: pd.DataFrame | None = None
    if not args.skip_confirmation:
        if not completed_trials(study):
            raise TrainingFailure("No successful screening trials to confirm")
        confirmations = confirmation_runs(
            study, args, shared, study_dir, output_root
        )
        confirmation_summary = save_confirmation_summary(confirmations, study_dir)
    write_report(study, study_dir, confirmation_summary)
    print(f"\nFinished. Review {study_dir / 'report.md'}")


if __name__ == "__main__":
    main()
