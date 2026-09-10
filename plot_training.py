from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MentalBERT + RDINO training history")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    history_path = args.output_dir / "training_history.csv"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Training history not found: {history_path}. "
            "It is created by training runs using the updated train.py."
        )
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError(f"Training history is empty: {history_path}")

    epochs = history["epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].plot(epochs, history["train_total_loss"], marker="o", label="Train")
    axes[0, 0].plot(
        epochs, history["validation_total_loss"], marker="o", label="Validation"
    )
    axes[0, 0].set(title="Combined loss", xlabel="Epoch", ylabel="Loss")
    loss_lines = axes[0, 0].lines
    if "learning_rate" in history.columns:
        learning_rate_axis = axes[0, 0].twinx()
        learning_rate_axis.plot(
            epochs,
            history["learning_rate"],
            color="tab:green",
            linestyle=":",
            marker="x",
            label="Learning rate",
        )
        learning_rate_axis.set_ylabel("Learning rate")
        learning_rate_axis.set_yscale("log")
        loss_lines += learning_rate_axis.lines
    axes[0, 0].legend(loss_lines, [line.get_label() for line in loss_lines])

    for split, style in (("train", "-"), ("validation", "--")):
        axes[0, 1].plot(
            epochs,
            history[f"{split}_classification_loss"],
            linestyle=style,
            marker="o",
            label=f"{split.title()} classification",
        )
        axes[0, 1].plot(
            epochs,
            history[f"{split}_regression_loss"],
            linestyle=style,
            marker="o",
            label=f"{split.title()} regression",
        )
    axes[0, 1].set(title="Loss components", xlabel="Epoch", ylabel="Loss")
    axes[0, 1].legend(fontsize="small")

    for column, label in (
        ("validation_accuracy", "Accuracy"),
        ("validation_balanced_accuracy", "Balanced accuracy"),
        ("validation_macro_f1", "Macro F1"),
    ):
        axes[1, 0].plot(epochs, history[column], marker="o", label=label)
    axes[1, 0].set(title="Validation classification", xlabel="Epoch", ylabel="Score")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    regression_axis = axes[1, 1]
    if "validation_regression_rmse" in history:
        validation_rmse = history["validation_regression_rmse"]
    elif "validation_regression_mse" in history:
        validation_rmse = history["validation_regression_mse"].pow(0.5)
    else:
        raise ValueError("Training history has no validation RMSE or MSE column")
    regression_axis.plot(
        epochs,
        validation_rmse,
        color="tab:blue",
        marker="o",
        label="Standardized RMSE",
    )
    regression_axis.set(
        title="Validation regression", xlabel="Epoch", ylabel="Standardized RMSE"
    )
    r2_axis = regression_axis.twinx()
    r2_axis.plot(
        epochs,
        history["validation_regression_r2"],
        color="tab:orange",
        marker="s",
        label="R²",
    )
    r2_axis.set_ylabel("R²")
    lines = regression_axis.lines + r2_axis.lines
    regression_axis.legend(lines, [line.get_label() for line in lines], loc="best")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    destination = args.output or args.output_dir / "training_curves.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
