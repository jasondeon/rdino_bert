"""Plot mixed-study predictions in the layout used for the LOSO report."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mixed_evaluation import evaluate_mixed
from scripts.score_evaluation_bundle import read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/mentalbert-mixed-20260913'))
    parser.add_argument('--bundle', type=Path, default=Path('outputs/evaluation/20260913_v3'))
    parser.add_argument('--reference-run', type=Path, default=Path('outputs/mentalbert-loso-20260913'))
    args = parser.parse_args()
    metrics, rows = evaluate_mixed(args.bundle, read_csv(args.run/'mixed_window_predictions.csv'))
    frame = pd.DataFrame(rows)
    reference = pd.read_csv(args.reference_run/'loso_recording_predictions.csv')
    # Use exactly the scatter limits in the LOSO plotting script.
    bottom = min(-5, 5*np.floor(reference.prediction.min()/5))
    top = 55
    rng = np.random.default_rng(40)
    summaries = []
    for entry in metrics['by_study']:
        group = frame[frame.study == entry['study']].copy()
        group['error'] = group.prediction-group.truth
        clusters = group.groupby('subject_id').error.agg(['sum', 'count']).to_numpy()
        draws = rng.multinomial(len(clusters), np.full(len(clusters), 1/len(clusters)), size=5000)
        sums = draws @ clusters
        low, high = np.percentile(sums[:,0]/sums[:,1], [2.5,97.5])
        summaries.append({'study': entry['study'], **entry['model'], 'mean_error_ci_low': low, 'mean_error_ci_high': high})
    summary = pd.DataFrame(summaries)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(zip(summary.study, ['#2563eb','#d97706','#7c3aed','#059669','#dc2626']))
    fig, axes = plt.subplots(2,3,figsize=(14,8.5),constrained_layout=True)
    for ax, row in zip(axes.flat, summaries):
        group = frame[frame.study == row['study']]
        ax.scatter(group.truth, group.prediction, s=16, alpha=.48, color=colors[row['study']], edgecolors='none')
        ax.plot([0,55],[0,55],color='#444444',linestyle='--',linewidth=1)
        ax.set(xlim=(-5,55), ylim=(bottom,top), xlabel='Observed MADRS', ylabel='Predicted MADRS',
               title=f"{row['study']}  |  R² {row['r2']:.2f}, r {row['pearson_r']:.2f}\nMean error {row['mean_prediction_error']:+.1f} points")
    ax = axes.flat[-1]
    bias = summary.mean_prediction_error
    ax.errorbar(bias,np.arange(len(summary)),xerr=np.array([bias-summary.mean_error_ci_low,summary.mean_error_ci_high-bias]),fmt='o',color='#334155',capsize=4)
    ax.axvline(0,color='gray',linestyle='--')
    ax.set(yticks=np.arange(len(summary)),yticklabels=summary.study,xlabel='Mean prediction error (MADRS points)',title='Study bias with subject-bootstrap 95% intervals')
    ax.invert_yaxis()
    reference_summary = pd.read_csv(args.reference_run/'analysis_v2/study_results.csv')
    bias_low = min(0, reference_summary.mean_error_ci_low.min())
    bias_high = max(0, reference_summary.mean_error_ci_high.max())
    bias_margin = .05*(bias_high-bias_low)
    ax.set_xlim(bias_low-bias_margin, bias_high+bias_margin)
    fig.suptitle('MentalBERT mixed-study: new subjects from represented studies',fontsize=15)
    output = args.run/'analysis'
    output.mkdir(exist_ok=True)
    for extension in ('png','pdf'):
        fig.savefig(output/f'study_predictions.{extension}',dpi=170)
    plt.close(fig)
    summary.to_csv(output/'study_prediction_plot_metrics.csv',index=False)
    (output/'study_prediction_plot_metadata.json').write_text(json.dumps({
        'bootstrap_replicates':5000,'seed':40,'method':'Resample subjects within each study, retaining all visits.',
        'scope':'Intervals conditional on fitted models and observed studies; no training-seed or future-study uncertainty.',
        'scatter_y_limits':[bottom,top],'recordings':len(frame),'predictions':'Uncalibrated outer-test predictions, recomputed from windows.'
    },indent=2)+'\n')
    print(output/'study_predictions.png')


if __name__ == '__main__':
    main()
