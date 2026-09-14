"""Describe finished LOSO predictions without changing models or test scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.score_evaluation_bundle import evaluate, read_csv


def cluster_statistics(frame):
    rows = []
    for _, group in frame.groupby("subject_id", sort=True):
        y, p, baseline = (group[k].to_numpy(float) for k in ("truth", "prediction", "training_mean_baseline"))
        rows.append([len(y), y.sum(), p.sum(), (y*y).sum(), (p*p).sum(), (y*p).sum(),
                     np.abs(p-y).sum(), np.square(p-y).sum(), (p-y).sum(),
                     np.abs(baseline-y).sum(), np.square(baseline-y).sum()])
    return np.asarray(rows, float)


def metrics_from_sums(values):
    n, sy, sp, syy, spp, syp, sae, sse, se, base_sae, base_sse = values.T
    variance_y, variance_p = syy-sy*sy/n, spp-sp*sp/n
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = (syp-sy*sp/n)/np.sqrt(variance_y*variance_p)
        r2 = 1-sse/variance_y
    return {"mae": sae/n, "rmse": np.sqrt(sse/n), "r2": r2, "pearson_r": correlation,
            "mean_error": se/n, "mae_gain_over_baseline": (base_sae-sae)/n}


def interval(values):
    values = np.asarray(values)
    return np.nanpercentile(values, [2.5, 97.5]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/mentalbert-loso-20260913"))
    parser.add_argument("--bundle", type=Path, default=Path("outputs/evaluation/20260913_v3"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mentalbert-loso-20260913/analysis"))
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new analysis output directory")
    metrics, predictions = evaluate(args.bundle, read_csv(args.run/"loso_window_predictions.csv"), "window", "text")
    frame = pd.DataFrame(predictions)
    saved = pd.read_csv(args.run/"loso_recording_predictions.csv", dtype={"subject_id": str})
    merged = frame.merge(saved, on="recording_id", suffixes=("", "_saved"), validate="one_to_one")
    if len(merged) != len(frame) or not np.allclose(merged["prediction"], merged["prediction_saved"], atol=1e-10):
        raise ValueError("Saved recording predictions do not match rescored window predictions")
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(40)
    summaries, curves, bootstrap_by_study, bootstrap_sums = [], [], {}, []
    for entry in metrics["by_study"]:
        study, model = entry["study"], entry["model"]
        group = frame[frame.study == study]
        stats = cluster_statistics(group)
        draws = rng.multinomial(len(stats), np.full(len(stats), 1/len(stats)), size=args.bootstrap)
        sums = draws @ stats
        boot = metrics_from_sums(sums)
        bootstrap_by_study[study] = boot
        bootstrap_sums.append(sums)
        y, p = group.truth.to_numpy(), group.prediction.to_numpy()
        selection = json.loads((args.run/study/"epoch_selection.json").read_text())
        bias = float(np.mean(p-y))
        centered_mse = float(np.mean(np.square((p-p.mean())-(y-y.mean()))))
        # Descriptive decomposition only: removing test-study bias and fitting
        # its optimal affine map use held-out labels and are NOT deployable scores.
        row = {"study": study, "subjects": entry["subjects"], "recordings": len(group),
               "mae": model["mae"], "baseline_mae": entry["training_mean_baseline"]["mae"],
               "rmse": model["rmse"], "r2": model["r2"], "pearson_r": model["pearson_r"],
               "spearman_r": float(spearmanr(y,p).statistic),
               "mean_error": bias, "observed_mean": float(y.mean()), "predicted_mean": float(p.mean()),
               "predicted_sd": float(p.std()), "observed_sd": float(y.std()),
               "calibration_slope": model["calibration_slope"], "selected_epoch": selection["selected_epoch"],
               "mae_gain_over_baseline": entry["training_mean_baseline"]["mae"]-model["mae"],
               "squared_bias_fraction_of_mse": bias*bias/model["rmse"]**2,
               "diagnostic_offset_removed_r2": 1-centered_mse/float(y.var()),
               "diagnostic_affine_fitted_r2": model["pearson_r"]**2,
               "predictions_below_zero": int(np.sum(p<0)), "predictions_above_sixty": int(np.sum(p>60))}
        for name, values in boot.items():
            row[name+"_ci_low"], row[name+"_ci_high"] = interval(values)
        summaries.append(row)
        for point in selection["curve"]:
            curves.append({"study": study, **point, "selected_epoch": selection["selected_epoch"]})
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output/"study_results.csv", index=False)
    pooled_boot = metrics_from_sums(sum(bootstrap_sums))
    macro_boot = {name: np.mean([values[name] for values in bootstrap_by_study.values()], axis=0)
                  for name in ("mae", "rmse", "mae_gain_over_baseline")}
    uncertainty = {"replicates": args.bootstrap, "seed": 40,
                   "method": "Paired subject-cluster bootstrap within each study; every visit retained when its subject is drawn",
                   "scope": "Conditional on these five studies, fitted models, and selected seed; does not refit models or estimate variation across future studies",
                   "pooled_95_percentile_intervals": {k: interval(v) for k,v in pooled_boot.items()},
                   "macro_study_95_percentile_intervals": {k: interval(v) for k,v in macro_boot.items()}}
    (args.output/"uncertainty.json").write_text(json.dumps(uncertainty, indent=2)+"\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = dict(zip(summary.study, ["#2563eb", "#d97706", "#7c3aed", "#059669", "#dc2626"]))
    fig, axes = plt.subplots(2,3,figsize=(14,8.5),constrained_layout=True)
    for ax, row in zip(axes.flat, summaries):
        group = frame[frame.study == row["study"]]
        ax.scatter(group.truth,group.prediction,s=16,alpha=.48,color=colors[row["study"]],edgecolors="none")
        ax.plot([0,55],[0,55],color="#444444",linestyle="--",linewidth=1,label="Perfect prediction")
        ax.set(xlim=(-5,55),ylim=(min(-5, 5*np.floor(frame.prediction.min()/5)),55),xlabel="Observed MADRS",ylabel="Predicted MADRS",
               title=f"{row['study']}  |  R² {row['r2']:.2f}, r {row['pearson_r']:.2f}\nMean error {row['mean_error']:+.1f} points")
    ax = axes.flat[-1]
    position=np.arange(len(summary))
    lower=summary.mean_error-summary.mean_error_ci_low
    upper=summary.mean_error_ci_high-summary.mean_error
    ax.errorbar(summary.mean_error,position,xerr=np.array([lower,upper]),fmt="o",color="#334155",capsize=4)
    ax.axvline(0,color="gray",linestyle="--")
    ax.set(yticks=position,yticklabels=summary.study,xlabel="Mean prediction error (MADRS points)",title="Study bias with subject-bootstrap 95% intervals")
    ax.invert_yaxis()
    fig.suptitle("MentalBERT leave-one-study-out: positive association, uneven absolute accuracy",fontsize=15)
    fig.savefig(args.output/"study_predictions.png",dpi=170)
    fig.savefig(args.output/"study_predictions.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    width=.35;positions=np.arange(len(summary))
    axes[0].bar(positions-width/2,summary.baseline_mae,width,color="#cbd5e1",label="Outer-training mean")
    axes[0].bar(positions+width/2,summary.mae,width,color="#2563eb",label="MentalBERT")
    axes[0].set(xticks=positions,xticklabels=summary.study,ylabel="MAE in MADRS points",title="Lower error than the training-mean baseline in every study")
    axes[0].legend(frameon=False)
    for study in summary.study:
        group=pd.DataFrame(curves);group=group[group.study==study]
        axes[1].plot(group.epoch,group.macro_study_rmse,color=colors[study],label=study)
        selected=group[group.epoch==group.selected_epoch]
        axes[1].scatter(selected.epoch,selected.macro_study_rmse,color=colors[study],s=35)
    axes[1].set(xlabel="Training epoch",ylabel="Mean inner-study RMSE",title="Epoch selection uses only inner validation studies")
    axes[1].legend(frameon=False,ncol=2)
    fig.savefig(args.output/"baseline_and_learning_curves.png",dpi=170)
    fig.savefig(args.output/"baseline_and_learning_curves.pdf")
    plt.close(fig)

    lines=["# MentalBERT LOSO results", "", "All 891 recordings / 515 subjects are present. Window predictions were independently aggregated and rescored; they match the saved recording predictions.", "",
           f"Pooled MAE {metrics['pooled']['mae']:.3f}, RMSE {metrics['pooled']['rmse']:.3f}, R² {metrics['pooled']['r2']:.3f}.",
           f"Equal-study MAE {metrics['macro_study']['mae']:.3f} versus training-mean baseline {metrics['macro_study_training_mean_baseline']['mae']:.3f}.", "",
           "| Study | Subjects / recordings | MAE | Baseline MAE | R² | Pearson r (95% CI) | Mean error | Selected epoch |",
           "|---|---:|---:|---:|---:|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['study']} | {row['subjects']} / {row['recordings']} | {row['mae']:.2f} | {row['baseline_mae']:.2f} | {row['r2']:.3f} | {row['pearson_r']:.2f} ({row['pearson_r_ci_low']:.2f}–{row['pearson_r_ci_high']:.2f}) | {row['mean_error']:+.2f} | {row['selected_epoch']} |")
    lines.extend(["", "## Interpretation", "",
        "The predictions show positive within-study associations with MADRS in every unseen study and reduce MAE relative to each fold's training-mean baseline. This is evidence of some transferable predictive information; it does not establish causal or depression-specific biomarkers.", "",
        "Absolute accuracy remains population-dependent. FORBOW is strongly overpredicted; OPTD and TIDE are underpredicted. Negative study R² means worse squared error than that test study's observed mean, which is a retrospective reference, not the deployable training-mean baseline.", "",
        "Calibration slopes below one show that differences in predictions are too large relative to their observed association with outcomes. Removing a mean offset alone is insufficient, especially for FORBOW. The offset-removed and affine-fitted R² columns in study_results.csv use test labels: they are diagnostic decompositions, NOT corrected validation results or deployable performance.", "",
        "Pooled R² is not a direct comparison to the previous 0.32: both the holdout design and the cohort/window/aggregation protocol changed. Study-wise results matter more than their pooled average for an unseen-population deployment target.", "",
        "The bootstrap keeps repeated visits together and resamples within the five observed studies. Intervals condition on these trained models and this seed, do not include training or model-selection uncertainty, and do not establish performance on all future studies.", "",
        "Next: test a simple calibration mapping learned only from inner out-of-fold predictions, and obtain a matched in-mixture subject-disjoint reference before attributing the historical performance change specifically to study shift. Continue to treat existing outer folds as development evidence after inspecting them. Avoid launching a broad model search based on these results alone.", "",
        "![Study predictions](study_predictions.png)", "", "![Baseline and learning curves](baseline_and_learning_curves.png)"])
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print(summary.round(4).to_string(index=False))
    print(json.dumps(uncertainty,indent=2))


if __name__ == "__main__":
    main()
