from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_wavlm_domain_generalization import load_recordings
from analyze_wavlm_longitudinal_change import (
    Candidate,
    candidate_features,
    correlation,
    make_pairs,
    metrics,
)
from analyze_wavlm_longitudinal_selection_control import (
    recording_covariates,
    select_candidate,
)


@dataclass
class FoldModel:
    held_out: np.ndarray
    candidate: Candidate
    nuisance: LinearRegression
    scaler: StandardScaler
    ridge: Ridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare true longitudinal WavLM changes with non-overlapping "
            "same-visit split-half changes"
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
    parser.add_argument("--random-splits", type=int, default=200)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def nonoverlapping_indices(group: pd.DataFrame) -> np.ndarray:
    selected = []
    last_end = -np.inf
    for row in group.sort_values(["window_start", "window_end"]).itertuples():
        if float(row.window_start) >= last_end - 1e-8:
            selected.append(int(row.embedding_index))
            last_end = float(row.window_end)
    if len(selected) < 2:
        raise ValueError(f"Fewer than two non-overlapping windows for {group.iloc[0]['audio_path']}")
    return np.asarray(selected, dtype=int)


def recording_window_lookup(windows: pd.DataFrame, paths: set[str]) -> dict[str, np.ndarray]:
    selected = windows[windows["audio_path"].isin(paths)]
    lookup = {
        str(path): nonoverlapping_indices(group)
        for path, group in selected.groupby("audio_path", sort=False)
    }
    missing = paths - set(lookup)
    if missing:
        raise ValueError(f"Missing cached windows for {sorted(missing)[:3]}")
    return lookup


def split_indices(
    indices: np.ndarray,
    mode: str,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(indices) // 2
    if mode == "contiguous":
        return indices[:count], indices[-count:]
    if mode == "random":
        if rng is None:
            raise ValueError("Random splitting requires an RNG")
        order = rng.permutation(indices)
        return order[:count], order[count : 2 * count]
    raise ValueError(mode)


def pseudo_features(
    paths: list[str],
    lookup: dict[str, np.ndarray],
    window_means: np.ndarray,
    window_stds: np.ndarray,
    layers: list[int],
    poolings: list[str],
    mode: str,
    rng: np.random.Generator | None,
) -> dict[tuple[int, str], np.ndarray]:
    mean_changes = {layer: [] for layer in layers}
    std_changes = {layer: [] for layer in layers}
    for path in paths:
        first, second = split_indices(lookup[path], mode, rng)
        for layer in layers:
            mean_changes[layer].append(
                np.asarray(window_means[second, layer], dtype=np.float32).mean(axis=0)
                - np.asarray(window_means[first, layer], dtype=np.float32).mean(axis=0)
            )
            std_changes[layer].append(
                np.asarray(window_stds[second, layer], dtype=np.float32).mean(axis=0)
                - np.asarray(window_stds[first, layer], dtype=np.float32).mean(axis=0)
            )
    result = {}
    for layer in layers:
        mean_delta = np.asarray(mean_changes[layer], dtype=np.float64)
        std_delta = np.asarray(std_changes[layer], dtype=np.float64)
        for pooling in poolings:
            result[(layer, pooling)] = (
                mean_delta
                if pooling == "mean"
                else np.concatenate([mean_delta, std_delta], axis=1)
            )
    return result


def fit_fold_models(
    feature_sets: dict[tuple[int, str], np.ndarray],
    pairs: pd.DataFrame,
    candidates: list[Candidate],
    inner_site_folds: int,
) -> tuple[list[FoldModel], np.ndarray, np.ndarray, np.ndarray]:
    truth = pairs["delta_madrs"].to_numpy(float)
    covariates = recording_covariates(pairs)
    sites = pairs["site"].to_numpy(str)
    total_prediction = np.full(len(pairs), np.nan)
    effect = np.full(len(pairs), np.nan)
    standardized_norm = np.full(len(pairs), np.nan)
    models = []
    splitter = GroupKFold(n_splits=len(np.unique(sites)))
    for fit, held_out in splitter.split(covariates, groups=sites):
        candidate = select_candidate(
            feature_sets,
            truth,
            covariates,
            sites,
            fit,
            candidates,
            inner_site_folds,
        )
        features = feature_sets[(candidate.layer, candidate.pooling)]
        nuisance = LinearRegression().fit(covariates[fit], truth[fit])
        residual = truth[fit] - nuisance.predict(covariates[fit])
        scaler = StandardScaler().fit(features[fit])
        ridge = Ridge(alpha=candidate.alpha).fit(
            scaler.transform(features[fit]), residual
        )
        held_features = features[held_out]
        embedding_prediction = ridge.predict(scaler.transform(held_features))
        zero_prediction = float(
            ridge.predict(scaler.transform(np.zeros((1, held_features.shape[1]))))[0]
        )
        total_prediction[held_out] = (
            nuisance.predict(covariates[held_out]) + embedding_prediction
        )
        effect[held_out] = embedding_prediction - zero_prediction
        standardized = scaler.transform(held_features) - scaler.transform(
            np.zeros_like(held_features)
        )
        standardized_norm[held_out] = np.linalg.norm(standardized, axis=1) / np.sqrt(
            standardized.shape[1]
        )
        models.append(FoldModel(held_out, candidate, nuisance, scaler, ridge))
        held_site = str(np.unique(sites[held_out])[0])
        print(f"held_site={held_site} selected=({candidate.name})", flush=True)
    return models, total_prediction, effect, standardized_norm


def apply_fold_models(
    models: list[FoldModel],
    feature_sets: dict[tuple[int, str], np.ndarray],
    pairs: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    covariates = recording_covariates(pairs)
    total_prediction = np.full(len(pairs), np.nan)
    effect = np.full(len(pairs), np.nan)
    standardized_norm = np.full(len(pairs), np.nan)
    for fold in models:
        features = feature_sets[(fold.candidate.layer, fold.candidate.pooling)][fold.held_out]
        transformed = fold.scaler.transform(features)
        embedding_prediction = fold.ridge.predict(transformed)
        zero_prediction = float(
            fold.ridge.predict(
                fold.scaler.transform(np.zeros((1, features.shape[1])))
            )[0]
        )
        total_prediction[fold.held_out] = (
            fold.nuisance.predict(covariates[fold.held_out]) + embedding_prediction
        )
        effect[fold.held_out] = embedding_prediction - zero_prediction
        centered = transformed - fold.scaler.transform(np.zeros_like(features))
        standardized_norm[fold.held_out] = np.linalg.norm(centered, axis=1) / np.sqrt(
            centered.shape[1]
        )
    return total_prediction, effect, standardized_norm


def summarize_pseudo(
    kind: str,
    repeat: int,
    truth: np.ndarray,
    prediction: np.ndarray,
    effect: np.ndarray,
    norm: np.ndarray,
) -> dict[str, float | int | str]:
    values = metrics(truth, prediction)
    return {
        "kind": kind,
        "repeat": repeat,
        **values,
        "effect_target_pearson": correlation(effect, truth),
        "effect_target_spearman": correlation(effect, truth, rank=True),
        "effect_absolute_median": float(np.median(np.abs(effect))),
        "standardized_feature_norm_median": float(np.median(norm)),
    }


def empirical_p(null: pd.Series, observed: float, larger_is_better: bool) -> float:
    extreme = null >= observed if larger_is_better else null <= observed
    return float((1 + extreme.sum()) / (len(null) + 1))


def plot_results(
    random_results: pd.DataFrame,
    observed: dict[str, float | int | str],
    contiguous: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = (
        ("r2", "R²", True),
        ("effect_target_pearson", "Embedding effect–ΔMADRS Pearson", True),
        ("standardized_feature_norm_median", "Median standardized change norm", True),
    )
    for axis, (column, label, _) in zip(axes, panels):
        for kind, color in (("baseline_random", "tab:blue"), ("week8_random", "tab:orange")):
            values = random_results.loc[random_results["kind"] == kind, column]
            axis.hist(values, bins=25, alpha=0.45, label=kind.replace("_", " "), color=color)
        axis.axvline(float(observed[column]), color="red", linewidth=2, label="true change")
        for _, row in contiguous.iterrows():
            axis.axvline(
                float(row[column]),
                color="black",
                linestyle="--",
                linewidth=1,
            )
        axis.set(xlabel=label, ylabel="Random splits")
    axes[0].legend()
    figure.suptitle("True longitudinal change versus same-visit split halves")
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
        "random_splits": args.random_splits,
        "window_selection": "greedy non-overlapping windows",
        "nuisance_covariates": [
            "baseline_madrs",
            "log1p_baseline_window_count",
            "log1p_week8_window_count",
        ],
        "seed": args.seed,
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; cached arrays were not loaded.")
        return

    recording_means, recording_stds, recordings = load_recordings(input_dir, "train")
    baseline_means, week8_means, baseline_stds, week8_stds, pairs = make_pairs(
        recording_means, recording_stds, recordings
    )
    candidates = [
        Candidate(layer, pooling, alpha)
        for layer in args.layers
        for pooling in args.poolings
        for alpha in args.ridge_alphas
    ]
    true_feature_sets = candidate_features(
        baseline_means,
        week8_means,
        baseline_stds,
        week8_stds,
        args.layers,
        args.poolings,
    )
    models, true_prediction, true_effect, true_norm = fit_fold_models(
        true_feature_sets, pairs, candidates, args.inner_site_folds
    )
    truth = pairs["delta_madrs"].to_numpy(float)
    true_summary = summarize_pseudo(
        "true_longitudinal", 0, truth, true_prediction, true_effect, true_norm
    )

    window_means = np.load(input_dir / "train_layer_means.npy", mmap_mode="r")
    window_stds = np.load(input_dir / "train_layer_stds.npy", mmap_mode="r")
    windows = pd.read_csv(input_dir / "train_windows.csv", dtype={"subject_id": str})
    baseline_paths = pairs["baseline_audio_path"].astype(str).tolist()
    week8_paths = pairs["week8_audio_path"].astype(str).tolist()
    lookup = recording_window_lookup(windows, set(baseline_paths + week8_paths))
    selected_layers = sorted({model.candidate.layer for model in models})
    selected_poolings = sorted({model.candidate.pooling for model in models})

    contiguous_rows = []
    contiguous_subject_frames = [
        pd.DataFrame(
            {
                "subject_id": pairs["subject_id"],
                "visit": "true_longitudinal",
                "effect": true_effect,
                "standardized_feature_norm": true_norm,
            }
        )
    ]
    for visit, paths in (("baseline", baseline_paths), ("week8", week8_paths)):
        features = pseudo_features(
            paths,
            lookup,
            window_means,
            window_stds,
            selected_layers,
            selected_poolings,
            "contiguous",
            None,
        )
        prediction, effect, norm = apply_fold_models(models, features, pairs)
        contiguous_rows.append(
            summarize_pseudo(
                f"{visit}_contiguous", 0, truth, prediction, effect, norm
            )
        )
        contiguous_subject_frames.append(
            pd.DataFrame(
                {
                    "subject_id": pairs["subject_id"],
                    "visit": visit,
                    "effect": effect,
                    "standardized_feature_norm": norm,
                }
            )
        )
    contiguous = pd.DataFrame(contiguous_rows)

    random_rows = []
    for repeat in range(1, args.random_splits + 1):
        for visit_index, (visit, paths) in enumerate(
            (("baseline", baseline_paths), ("week8", week8_paths))
        ):
            rng = np.random.default_rng(args.seed + 2 * repeat + visit_index)
            features = pseudo_features(
                paths,
                lookup,
                window_means,
                window_stds,
                selected_layers,
                selected_poolings,
                "random",
                rng,
            )
            prediction, effect, norm = apply_fold_models(models, features, pairs)
            random_rows.append(
                summarize_pseudo(
                    f"{visit}_random", repeat, truth, prediction, effect, norm
                )
            )
        if repeat % 10 == 0 or repeat == args.random_splits:
            print(f"completed random split {repeat}/{args.random_splits}", flush=True)
    random_results = pd.DataFrame(random_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    random_results.to_csv(output_dir / "random_split_metrics.csv", index=False)
    contiguous.to_csv(output_dir / "contiguous_split_metrics.csv", index=False)
    subject_effects = pd.concat(contiguous_subject_frames, ignore_index=True)
    subject_effects = subject_effects.merge(
        pairs[["subject_id", "site", "delta_madrs"]], on="subject_id", how="left"
    )
    subject_effects.to_csv(output_dir / "contiguous_subject_effects.csv", index=False)
    plot_results(
        random_results,
        true_summary,
        contiguous,
        output_dir / "split_half_stability.png",
    )

    p_values = {}
    for kind in ("baseline_random", "week8_random"):
        rows = random_results[random_results["kind"] == kind]
        p_values[kind] = {
            "r2": empirical_p(rows["r2"], float(true_summary["r2"]), True),
            "effect_target_pearson": empirical_p(
                rows["effect_target_pearson"],
                float(true_summary["effect_target_pearson"]),
                True,
            ),
            "standardized_feature_norm": empirical_p(
                rows["standardized_feature_norm_median"],
                float(true_summary["standardized_feature_norm_median"]),
                True,
            ),
        }
    summary = {
        "true_longitudinal": true_summary,
        "contiguous_same_visit": contiguous.to_dict("records"),
        "random_split_p_values": p_values,
        "selection_counts": Counter(model.candidate.name for model in models).most_common(),
        "nonoverlapping_windows_per_recording": {
            "minimum": int(min(len(value) for value in lookup.values())),
            "median": float(np.median([len(value) for value in lookup.values()])),
            "maximum": int(max(len(value) for value in lookup.values())),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# WavLM same-visit split-half stability audit",
        "",
        f"Subjects: **{len(pairs)}**. Random non-overlapping split repetitions: "
        f"**{args.random_splits}**.",
        "The clinical model is selected and fitted only with true longitudinal",
        "changes in leave-one-site-out folds. Same-visit changes are then passed",
        "through those held-out models without refitting.",
        "",
        "| Change source | R² | Effect–ΔMADRS Pearson | Median absolute effect | Median standardized change norm |",
        "|---|---:|---:|---:|---:|",
        f"| True week 8 − baseline | {true_summary['r2']:.3f} | "
        f"{true_summary['effect_target_pearson']:.3f} | "
        f"{true_summary['effect_absolute_median']:.3f} | "
        f"{true_summary['standardized_feature_norm_median']:.3f} |",
    ]
    for row in contiguous.to_dict("records"):
        lines.append(
            f"| {row['kind']} | {row['r2']:.3f} | "
            f"{row['effect_target_pearson']:.3f} | "
            f"{row['effect_absolute_median']:.3f} | "
            f"{row['standardized_feature_norm_median']:.3f} |"
        )
    lines.extend(["", "## Random non-overlapping halves", ""])
    for kind in ("baseline_random", "week8_random"):
        rows = random_results[random_results["kind"] == kind]
        lines.append(
            f"- **{kind}:** mean R² {rows['r2'].mean():.3f} "
            f"[{rows['r2'].quantile(.025):.3f}, {rows['r2'].quantile(.975):.3f}], "
            f"p={p_values[kind]['r2']:.4f}; mean effect–target Pearson "
            f"{rows['effect_target_pearson'].mean():.3f} "
            f"[{rows['effect_target_pearson'].quantile(.025):.3f}, "
            f"{rows['effect_target_pearson'].quantile(.975):.3f}], "
            f"p={p_values[kind]['effect_target_pearson']:.4f}; mean median-norm "
            f"{rows['standardized_feature_norm_median'].mean():.3f} "
            f"[{rows['standardized_feature_norm_median'].quantile(.025):.3f}, "
            f"{rows['standardized_feature_norm_median'].quantile(.975):.3f}], "
            f"p={p_values[kind]['standardized_feature_norm']:.4f}."
        )
    lines.extend(
        [
            "",
            f"Selected models: `{json.dumps(summary['selection_counts'])}`.",
            "",
            "If true longitudinal changes exceed both random and contiguous",
            "same-visit changes, finite-window sampling and ordinary within-session",
            "drift are unlikely to explain the signal. This audit still cannot",
            "separate clinical change from other between-visit session changes.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
