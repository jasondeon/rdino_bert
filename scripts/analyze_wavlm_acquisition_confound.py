from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dataset import MultimodalDataset
from analyze_wavlm_domain_generalization import load_recordings
from analyze_wavlm_longitudinal_change import (
    Candidate,
    candidate_features,
    make_pairs,
    metrics,
    paired_bootstrap,
)
from analyze_wavlm_longitudinal_selection_control import (
    evaluate_embedding_loso,
    recording_covariates,
)
from analyze_wavlm_split_half_stability import recording_window_lookup


WINDOW_STATISTICS = (
    "raw_rms_db",
    "raw_peak_db",
    "raw_dc_relative",
    "clipping_log10_fraction",
    "normalized_crest_db",
    "relative_noise_floor_db",
    "frame_dynamic_range_db",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "low_frequency_ratio",
    "high_frequency_ratio",
    "spectral_flatness",
)

SURVIVING_CHANGE_COLUMNS = (
    "delta_clipping_log10_fraction",
    "delta_normalized_crest_db",
    "delta_relative_noise_floor_db",
    "delta_frame_dynamic_range_db",
    "delta_zero_crossing_rate",
    "delta_spectral_centroid_hz",
    "delta_low_frequency_ratio",
    "delta_high_frequency_ratio",
    "delta_spectral_flatness",
    "sample_rate_changed",
    "channels_changed",
    "subtype_changed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether technical acquisition changes explain longitudinal "
            "WavLM prediction"
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layers", required=True, type=int, nargs="+")
    parser.add_argument(
        "--poolings",
        choices=("mean", "mean_temporal_std"),
        nargs="+",
        default=("mean", "mean_temporal_std"),
    )
    parser.add_argument(
        "--ridge-alphas",
        type=float,
        nargs="+",
        default=(100.0, 1000.0, 10000.0, 100000.0),
    )
    parser.add_argument("--inner-site-folds", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def finite_db(value: float, multiplier: float = 20.0) -> float:
    return float(multiplier * np.log10(max(value, 1e-12)))


def waveform_statistics(waveform: np.ndarray, sample_rate: int) -> dict[str, float]:
    values = np.asarray(waveform, dtype=np.float64)
    mean = float(values.mean())
    centered = values - mean
    rms = float(np.sqrt(np.mean(values**2)))
    centered_rms = float(np.sqrt(np.mean(centered**2)))
    peak = float(np.max(np.abs(values)))
    normalized = centered / np.sqrt(np.var(values) + 1e-7)

    frame_length = max(1, sample_rate // 10)
    frame_count = len(normalized) // frame_length
    frames = normalized[: frame_count * frame_length].reshape(frame_count, frame_length)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    noise_floor = float(np.quantile(frame_rms, 0.1))
    frame_high = float(np.quantile(frame_rms, 0.9))

    frequencies, power = welch(
        normalized,
        fs=sample_rate,
        window="hann",
        nperseg=512,
        noverlap=256,
        detrend=False,
        scaling="spectrum",
    )
    power = np.maximum(power, 1e-20)
    total_power = float(power.sum())
    centroid = float(np.sum(frequencies * power) / total_power)
    low_ratio = float(power[frequencies < 200].sum() / total_power)
    high_ratio = float(power[frequencies >= 4000].sum() / total_power)
    flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
    return {
        "raw_rms_db": finite_db(rms),
        "raw_peak_db": finite_db(peak),
        "raw_dc_relative": abs(mean) / max(centered_rms, 1e-12),
        "clipping_log10_fraction": float(
            np.log10(np.mean(np.abs(values) >= 0.999) + 1e-7)
        ),
        "normalized_crest_db": finite_db(
            float(np.max(np.abs(normalized)))
            / max(float(np.sqrt(np.mean(normalized**2))), 1e-12)
        ),
        "relative_noise_floor_db": finite_db(noise_floor),
        "frame_dynamic_range_db": finite_db(frame_high / max(noise_floor, 1e-12)),
        "zero_crossing_rate": float(np.mean(np.signbit(normalized[1:]) != np.signbit(normalized[:-1]))),
        "spectral_centroid_hz": centroid,
        "low_frequency_ratio": low_ratio,
        "high_frequency_ratio": high_ratio,
        "spectral_flatness": flatness,
    }


def make_dataset(config: dict[str, object]) -> MultimodalDataset:
    return MultimodalDataset(
        Path(str(config["train"])),
        sample_rate=16_000,
        window_seconds=float(config["window_seconds"]),
        stride_seconds=float(config["stride_seconds"]),
        num_classes=4,
        include_empty_text=bool(config["include_empty_text"]),
        load_audio=True,
        eligibility_window_seconds=float(config["eligibility_window_seconds"]),
        speaker_gap_policy=str(config["speaker_gap_policy"]),
    )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def extract_recording_statistics(
    input_dir: Path,
    output_path: Path,
    extraction: dict[str, object],
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    windows = pd.read_csv(input_dir / "train_windows.csv", dtype={"subject_id": str})
    paths = set(pairs["baseline_audio_path"].astype(str)) | set(
        pairs["week8_audio_path"].astype(str)
    )
    lookup = recording_window_lookup(windows, paths)
    dataset = make_dataset(extraction)
    if len(dataset) != len(windows):
        raise ValueError(
            f"Reconstructed dataset has {len(dataset)} windows; cache has {len(windows)}"
        )
    if output_path.exists():
        rows = pd.read_csv(output_path).to_dict("records")
        completed = {str(row["audio_path"]) for row in rows}
        print(f"Resuming after {len(completed)} recordings", flush=True)
    else:
        rows = []
        completed = set()
    for recording_number, audio_path in enumerate(sorted(paths), start=1):
        if audio_path in completed:
            continue
        info = sf.info(audio_path)
        per_window = []
        for embedding_index in lookup[audio_path]:
            item = dataset[int(embedding_index)]
            if str(item["audio_path"]) != audio_path:
                raise ValueError("Dataset/cache window ordering does not match")
            per_window.append(
                waveform_statistics(item["waveform"].numpy(), sample_rate=16_000)
            )
        frame = pd.DataFrame(per_window)
        row: dict[str, object] = {
            "audio_path": audio_path,
            "original_sample_rate": int(info.samplerate),
            "original_channels": int(info.channels),
            "original_format": str(info.format),
            "original_subtype": str(info.subtype),
            "file_duration_seconds": float(info.frames / info.samplerate),
            "nonoverlapping_windows": len(per_window),
        }
        for column in WINDOW_STATISTICS:
            row[column] = float(frame[column].median())
            row[f"{column}_window_iqr"] = float(
                frame[column].quantile(0.75) - frame[column].quantile(0.25)
            )
        rows.append(row)
        if recording_number % 10 == 0 or recording_number == len(paths):
            atomic_write_csv(pd.DataFrame(rows), output_path)
            print(
                f"audio statistics: {recording_number}/{len(paths)} recordings",
                flush=True,
            )
    result = pd.DataFrame(rows)
    if len(result) != len(paths):
        raise RuntimeError(f"Expected {len(paths)} recordings, found {len(result)}")
    return result


def make_change_features(
    recordings: pd.DataFrame, pairs: pd.DataFrame
) -> pd.DataFrame:
    indexed = recordings.set_index("audio_path")
    rows = []
    for pair in pairs.itertuples():
        baseline = indexed.loc[str(pair.baseline_audio_path)]
        week8 = indexed.loc[str(pair.week8_audio_path)]
        row: dict[str, object] = {
            "subject_id": pair.subject_id,
            "site": pair.site,
            "delta_madrs": float(pair.delta_madrs),
        }
        for column in WINDOW_STATISTICS:
            row[f"delta_{column}"] = float(week8[column] - baseline[column])
        row["sample_rate_changed"] = float(
            week8["original_sample_rate"] != baseline["original_sample_rate"]
        )
        row["channels_changed"] = float(
            week8["original_channels"] != baseline["original_channels"]
        )
        row["format_changed"] = float(
            week8["original_format"] != baseline["original_format"]
        )
        row["subtype_changed"] = float(
            week8["original_subtype"] != baseline["original_subtype"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_linear_loso(features: np.ndarray, pairs: pd.DataFrame) -> np.ndarray:
    truth = pairs["delta_madrs"].to_numpy(float)
    sites = pairs["site"].to_numpy(str)
    prediction = np.full(len(pairs), np.nan)
    splitter = GroupKFold(n_splits=len(np.unique(sites)))
    for fit, held_out in splitter.split(features, groups=sites):
        model = LinearRegression().fit(features[fit], truth[fit])
        prediction[held_out] = model.predict(features[held_out])
    return prediction


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def correlation_table(changes: pd.DataFrame, wavlm_effect: np.ndarray) -> pd.DataFrame:
    rows = []
    truth = changes["delta_madrs"].to_numpy(float)
    for column in [
        name for name in changes
        if name.startswith("delta_") and name != "delta_madrs"
    ] + [
        "sample_rate_changed",
        "channels_changed",
        "format_changed",
        "subtype_changed",
    ]:
        values = changes[column].to_numpy(float)
        if np.std(values) < 1e-12:
            outcome_rho = effect_rho = 0.0
            outcome_p = effect_p = 1.0
        else:
            outcome = spearmanr(values, truth)
            effect = spearmanr(values, wavlm_effect)
            outcome_rho, outcome_p = float(outcome.statistic), float(outcome.pvalue)
            effect_rho, effect_p = float(effect.statistic), float(effect.pvalue)
        rows.append(
            {
                "feature": column,
                "delta_madrs_spearman": outcome_rho,
                "delta_madrs_p": outcome_p,
                "wavlm_effect_spearman": effect_rho,
                "wavlm_effect_p": effect_p,
            }
        )
    result = pd.DataFrame(rows)
    result["delta_madrs_fdr"] = benjamini_hochberg(
        result["delta_madrs_p"].to_numpy(float)
    )
    result["wavlm_effect_fdr"] = benjamini_hochberg(
        result["wavlm_effect_p"].to_numpy(float)
    )
    return result.sort_values("wavlm_effect_p")


def plot_results(
    model_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].bar(
        np.arange(len(model_summary)),
        model_summary["r2"],
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(
        np.arange(len(model_summary)),
        ["Baseline", "+windows", "+acquisition", "+WavLM"],
        rotation=20,
    )
    axes[0].set(title="Leave-one-site-out prediction", ylabel="R²")
    shown = correlations.head(10).sort_values("wavlm_effect_spearman")
    axes[1].barh(shown["feature"], shown["wavlm_effect_spearman"])
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set(title="Technical-change association with WavLM effect", xlabel="Spearman")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    extraction = json.loads((input_dir / "extraction_config.json").read_text())
    layer_names = list(extraction["layer_names"])
    if any(layer < 0 or layer >= len(layer_names) for layer in args.layers):
        raise ValueError(f"Layers must be between 0 and {len(layer_names) - 1}")
    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model_name": extraction["model_name"],
        "window_seconds": extraction["window_seconds"],
        "stride_seconds": extraction["stride_seconds"],
        "layers": args.layers,
        "poolings": args.poolings,
        "ridge_alphas": args.ridge_alphas,
        "inner_site_folds": args.inner_site_folds,
        "bootstrap_samples": args.bootstrap_samples,
        "technical_features": list(SURVIVING_CHANGE_COLUMNS),
        "waveform_source": "exact cached non-overlapping dataset windows",
        "wavlm_window_normalization": "zero mean, unit variance",
        "seed": args.seed,
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; audio was not loaded.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Existing configuration differs: {config_path}")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    means, stds, recording_metadata = load_recordings(input_dir, "train")
    baseline_means, week8_means, baseline_stds, week8_stds, pairs = make_pairs(
        means, stds, recording_metadata
    )
    recording_stats = extract_recording_statistics(
        input_dir,
        output_dir / "recording_acquisition_features.csv",
        extraction,
        pairs,
    )
    changes = make_change_features(recording_stats, pairs)
    changes.to_csv(output_dir / "paired_acquisition_changes.csv", index=False)

    window_covariates = recording_covariates(pairs)
    technical = changes[list(SURVIVING_CHANGE_COLUMNS)].to_numpy(float)
    all_covariates = np.column_stack([window_covariates, technical])
    baseline_prediction = evaluate_linear_loso(window_covariates[:, :1], pairs)
    window_prediction = evaluate_linear_loso(window_covariates, pairs)
    technical_prediction = evaluate_linear_loso(all_covariates, pairs)

    feature_sets = candidate_features(
        baseline_means,
        week8_means,
        baseline_stds,
        week8_stds,
        args.layers,
        args.poolings,
    )
    candidates = [
        Candidate(layer, pooling, alpha)
        for layer in args.layers
        for pooling in args.poolings
        for alpha in args.ridge_alphas
    ]
    wavlm_prediction, wavlm_selections = evaluate_embedding_loso(
        feature_sets,
        pairs,
        candidates,
        args.inner_site_folds,
        verbose=True,
        covariates=all_covariates,
    )
    truth = pairs["delta_madrs"].to_numpy(float)
    predictions = pairs[["subject_id", "site", "delta_madrs"]].copy()
    predictions["baseline_madrs"] = baseline_prediction
    predictions["baseline_plus_window_counts"] = window_prediction
    predictions["baseline_plus_acquisition"] = technical_prediction
    predictions["baseline_plus_acquisition_plus_wavlm"] = wavlm_prediction
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    model_rows = []
    for name in predictions.columns[3:]:
        model_rows.append({"model": name, **metrics(truth, predictions[name].to_numpy(float))})
    model_summary = pd.DataFrame(model_rows)
    model_summary.to_csv(output_dir / "model_summary.csv", index=False)
    wavlm_effect = wavlm_prediction - technical_prediction
    correlations = correlation_table(changes, wavlm_effect)
    correlations.to_csv(output_dir / "feature_correlations.csv", index=False)
    bootstrap = paired_bootstrap(
        truth,
        technical_prediction,
        wavlm_prediction,
        args.bootstrap_samples,
        args.seed,
    )
    plot_results(
        model_summary,
        correlations,
        output_dir / "acquisition_confound_audit.png",
    )

    values = model_summary.set_index("model")
    best_correlations = correlations.head(8)
    lines = [
        "# WavLM technical acquisition-confound audit",
        "",
        f"Paired subjects: **{len(pairs)}** across **{pairs['site'].nunique()}** sites.",
        "Audio measurements use the exact non-overlapping waveforms supplied to",
        "WavLM after mono conversion and 16-kHz resampling. WavLM then applies",
        "per-window zero-mean/unit-variance normalization, so raw gain and DC",
        "offset are reported but excluded from the primary adjustment model.",
        "",
        "| LOSO model | R² | RMSE | Pearson |",
        "|---|---:|---:|---:|",
    ]
    for model in model_summary["model"]:
        row = values.loc[model]
        lines.append(
            f"| {model} | {row['r2']:.3f} | {row['rmse']:.3f} | {row['pearson']:.3f} |"
        )
    acquisition = values.loc["baseline_plus_acquisition"]
    wavlm = values.loc["baseline_plus_acquisition_plus_wavlm"]
    lines.extend(
        [
            "",
            f"WavLM beyond acquisition controls: ΔR² {wavlm['r2'] - acquisition['r2']:.3f} "
            f"[{bootstrap['r2_difference_ci_low']:.3f}, "
            f"{bootstrap['r2_difference_ci_high']:.3f}], ΔRMSE "
            f"{wavlm['rmse'] - acquisition['rmse']:.3f} "
            f"[{bootstrap['rmse_difference_ci_low']:.3f}, "
            f"{bootstrap['rmse_difference_ci_high']:.3f}].",
            "",
            f"WavLM selections: `{json.dumps(Counter(wavlm_selections).most_common())}`.",
            "",
            "## Strongest technical associations with the WavLM contribution",
            "",
            "| Feature | Spearman | FDR q | ΔMADRS Spearman | ΔMADRS FDR q |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in best_correlations.to_dict("records"):
        lines.append(
            f"| {row['feature']} | {row['wavlm_effect_spearman']:.3f} | "
            f"{row['wavlm_effect_fdr']:.3f} | {row['delta_madrs_spearman']:.3f} | "
            f"{row['delta_madrs_fdr']:.3f} |"
        )
    lines.extend(
        [
            "",
            "If the WavLM gain disappears after adjustment, acquisition changes are",
            "a plausible explanation. If it persists and no technical feature tracks",
            "the WavLM contribution, these measured acquisition properties do not",
            "explain the longitudinal signal.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
