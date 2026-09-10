from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.importance import get_param_importances

from tune_hyperparameters import (
    PROJECT_ROOT,
    TrainingFailure,
    completed_trials,
    record_trial_metrics,
    resolved_from_project,
    run_training,
)


WINDOW_CHOICES = [20.0, 25.0, 30.0]
STRIDE_CHOICES = [10.0, 15.0, 20.0]
RANK_CHOICES = [1, 2, 4, 8]
ALPHA_CHOICES = [4, 8, 16, 32]
ACCUMULATION_CHOICES = [1, 2, 4]
WEIGHT_DECAY_CHOICES = [0.0, 1e-6, 1e-5, 1e-4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable MentalBERT-only regression HPO"
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
        "--rdino-checkpoint",
        type=Path,
        default=Path("assets/pretrained_rdino.pth"),
    )
    parser.add_argument(
        "--study-name", default="mentalbert-text-regression-hpo-v1"
    )
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("experiments/hpo_text_regression"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/hpo_text_regression"),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Target total number of resumable trials (default: 20)",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=10.0,
        help=(
            "Stop starting new trials after this many hours; the active trial "
            "is allowed to finish (default: 10, use 0 for no time limit)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-windows-per-recording", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--screening-seed", type=int, default=40)
    parser.add_argument("--sampler-seed", type=int, default=2027)
    parser.add_argument("--pruning-startup-trials", type=int, default=6)
    parser.add_argument(
        "--pruning-warmup-epoch",
        type=int,
        default=50,
        help="Do not prune a run before this training epoch (default: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the study configuration without training",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    if args.timeout_hours < 0:
        raise ValueError("--timeout-hours cannot be negative")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1")
    if args.validation_interval < 1:
        raise ValueError("--validation-interval must be at least 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.train_windows_per_recording < 1:
        raise ValueError("--train-windows-per-recording must be at least 1")
    if args.batch_size % args.train_windows_per_recording != 0:
        raise ValueError(
            "--batch-size must be divisible by --train-windows-per-recording"
        )
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.pruning_startup_trials < 1:
        raise ValueError("--pruning-startup-trials must be at least 1")
    if args.pruning_warmup_epoch < args.validation_interval:
        raise ValueError(
            "--pruning-warmup-epoch must be at least --validation-interval"
        )

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
        "validation_manifest": str(
            resolved_from_project(args.validation_manifest)
        ),
        "rdino_checkpoint": str(resolved_from_project(args.rdino_checkpoint)),
        "text_model": "mental/mental-bert-base-uncased",
        "modality": "text",
        "num_classes": 4,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "train_sampling": "recording",
        "train_windows_per_recording": args.train_windows_per_recording,
        "eligibility_window_seconds": max(WINDOW_CHOICES),
        "classification_weight": 0.0,
        "regression_weight": 1.0,
        "class_weighting": "none",
        "standardize_regression_labels": True,
        "lr_scheduler": "none",
        "early_stopping_patience": args.early_stopping_patience,
        "validation_interval": args.validation_interval,
        "seed": args.screening_seed,
    }


def suggested_parameters(trial: optuna.Trial) -> dict[str, Any]:
    disable_text_lora = trial.suggest_categorical(
        "disable_text_lora", [False, True]
    )
    parameters: dict[str, Any] = {
        "embedding_normalization": trial.suggest_categorical(
            "embedding_normalization", ["layernorm", "batchnorm"]
        ),
        "disable_text_lora": disable_text_lora,
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", ACCUMULATION_CHOICES
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 3e-6, 3e-5, log=True
        ),
        "weight_decay": trial.suggest_categorical(
            "weight_decay", WEIGHT_DECAY_CHOICES
        ),
        "window_seconds": trial.suggest_categorical(
            "window_seconds", WINDOW_CHOICES
        ),
        "stride_seconds": trial.suggest_categorical(
            "stride_seconds", STRIDE_CHOICES
        ),
    }
    if not disable_text_lora:
        parameters["lora_rank"] = trial.suggest_categorical(
            "lora_rank", RANK_CHOICES
        )
        parameters["lora_alpha"] = trial.suggest_categorical(
            "lora_alpha", ALPHA_CHOICES
        )
    return parameters


def save_artifacts(study: optuna.Study, study_dir: Path) -> None:
    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    importances: dict[str, float] = {}
    if len(completed_trials(study)) >= 4:
        try:
            importances = get_param_importances(study)
        except (RuntimeError, ValueError):
            pass
    (study_dir / "parameter_importance.json").write_text(
        json.dumps(importances, indent=2) + "\n", encoding="utf-8"
    )


def write_report(study: optuna.Study, study_dir: Path) -> None:
    ranked = completed_trials(study)
    lines = [
        "# MentalBERT-only regression hyperparameter search",
        "",
        "Objective: best-epoch, full-recording validation R-squared.",
        "Classification is disabled, recording-level sampling uses four windows,",
        "the LR scheduler is disabled, and all window choices use the same",
        "30-second-eligible recording cohort.",
        "",
        "| Trial | R2 | RMSE | Original RMSE | Best epoch | Parameters |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trial in ranked[:10]:
        rmse = trial.user_attrs.get("validation_regression_rmse", float("nan"))
        original_rmse = trial.user_attrs.get(
            "validation_regression_rmse_original_scale", float("nan")
        )
        best_epoch = trial.user_attrs.get("best_epoch", -1)
        lines.append(
            f"| {trial.number} | {float(trial.value):.6f} | {rmse:.6f} | "
            f"{original_rmse:.6f} | {best_epoch} | "
            f"`{json.dumps(trial.params, sort_keys=True)}` |"
        )
    if not ranked:
        lines.append("| - | - | - | - | - | No completed trials |")

    importance_path = study_dir / "parameter_importance.json"
    importances = json.loads(importance_path.read_text(encoding="utf-8"))
    lines.extend(["", "## Parameter importance", ""])
    if importances:
        for parameter, importance in importances.items():
            lines.append(f"- `{parameter}`: {importance:.4f}")
    else:
        lines.append("Not enough compatible completed trials to estimate reliably.")
    lines.extend(
        [
            "",
            "LoRA rank and alpha are sampled only when LoRA is enabled.",
            "Use the future untouched comparison set for the final model estimate;",
            "the best value here is optimized on the current validation set.",
        ]
    )
    (study_dir / "report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def describe(args: argparse.Namespace, shared: dict[str, Any]) -> None:
    print("Validated standalone MentalBERT regression HPO configuration.")
    print(f"Target trials: {args.n_trials}")
    print(f"Overnight limit: {args.timeout_hours:g} hours")
    print(
        f"Maximum epochs: {args.epochs}; validation every "
        f"{args.validation_interval} epochs"
    )
    print(
        f"Median pruning starts after {args.pruning_startup_trials} trials "
        f"and epoch {args.pruning_warmup_epoch}."
    )
    print("Fixed settings:")
    print(json.dumps(shared, indent=2, sort_keys=True))
    print("Search space:")
    print(
        json.dumps(
            {
                "embedding_normalization": ["layernorm", "batchnorm"],
                "disable_text_lora": [False, True],
                "lora_rank_when_enabled": RANK_CHOICES,
                "lora_alpha_when_enabled": ALPHA_CHOICES,
                "gradient_accumulation_steps": ACCUMULATION_CHOICES,
                "learning_rate": "log-uniform [3e-6, 3e-5]",
                "weight_decay": WEIGHT_DECAY_CHOICES,
                "window_seconds": WINDOW_CHOICES,
                "stride_seconds": STRIDE_CHOICES,
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
        describe(args, shared)
        return

    study_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    storage_path = study_dir / "study.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.sampler_seed,
            n_startup_trials=args.pruning_startup_trials,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=args.pruning_startup_trials,
            n_warmup_steps=args.pruning_warmup_epoch,
            interval_steps=args.validation_interval,
        ),
        load_if_exists=True,
    )

    if not study.trials:
        study.enqueue_trial(
            {
                "embedding_normalization": "layernorm",
                "disable_text_lora": False,
                "lora_rank": 2,
                "lora_alpha": 16,
                "gradient_accumulation_steps": 2,
                "learning_rate": 1e-5,
                "weight_decay": 1e-5,
                "window_seconds": 30.0,
                "stride_seconds": 20.0,
            }
        )

    def objective(trial: optuna.Trial) -> float:
        parameters = {**shared, **suggested_parameters(trial)}
        output_dir = output_root / "screening" / f"trial-{trial.number:04d}"
        best = run_training(
            f"text-regression-trial-{trial.number:04d}",
            parameters,
            output_dir,
            trial=trial,
        )
        record_trial_metrics(trial, best, output_dir)
        return float(best["validation_regression_r2"])

    remaining = max(0, args.n_trials - len(study.trials))
    if remaining:
        timeout = args.timeout_hours * 3600 if args.timeout_hours > 0 else None
        print(f"Running up to {remaining} trials to reach {args.n_trials} total.")
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=timeout,
            callbacks=[lambda current, _: save_artifacts(current, study_dir)],
            gc_after_trial=True,
            n_jobs=1,
            catch=(TrainingFailure,),
        )
    else:
        print(f"Study already has {len(study.trials)} trials.")

    save_artifacts(study, study_dir)
    write_report(study, study_dir)
    report_path = study_dir / "report.md"
    print(f"Finished. Review {report_path}")


if __name__ == "__main__":
    main()
