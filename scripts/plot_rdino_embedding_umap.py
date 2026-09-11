from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.metrics import silhouette_score


WINDOWS = (5.0, 10.0, 20.0, 30.0)
CLASS_NAMES = {
    0: "Class 0 (0–4)",
    1: "Class 1 (5–11)",
    2: "Class 2 (12–22)",
    3: "Class 3 (≥23)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot separate UMAPs of recording-level frozen RDINO embeddings across "
            "window sizes"
        )
    )
    parser.add_argument(
        "--audit-root", type=Path, default=Path("outputs/rdino-window-size-audit")
    )
    parser.add_argument(
        "--reference-20-dir",
        type=Path,
        default=Path("outputs/rdino-embedding-audit-preserved-gaps"),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
    )
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--min-dist", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=40)
    return parser.parse_args()


def directories(audit_root: Path, reference_20_dir: Path) -> dict[float, Path]:
    return {
        5.0: audit_root / "window-05",
        10.0: audit_root / "window-10",
        20.0: reference_20_dir,
        30.0: audit_root / "window-30",
    }


def aggregate_recordings(directory: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(directory / f"{split}_embeddings.npy", mmap_mode="r")
    windows = pd.read_csv(
        directory / f"{split}_windows.csv", dtype={"subject_id": str}
    )
    if embeddings.ndim != 2 or len(embeddings) != len(windows):
        raise ValueError(f"Misaligned embedding cache in {directory} for {split}")
    expected = np.arange(len(windows))
    if not np.array_equal(windows["embedding_index"].to_numpy(), expected):
        raise ValueError(f"Nonsequential embedding indices in {directory} for {split}")

    features: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for recording_index, group in windows.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(dtype=int)
        labels = group["class_label"].to_numpy(dtype=int)
        if not np.all(labels == labels[0]):
            raise ValueError(f"Class labels differ within recording {recording_index}")
        first = group.iloc[0]
        features.append(
            np.asarray(embeddings[indices], dtype=np.float64).mean(axis=0)
        )
        rows.append(
            {
                "recording_index": int(recording_index),
                "audio_path": str(first["audio_path"]),
                "subject_id": str(first["subject_id"]),
                "class_label": int(labels[0]),
                "window_count": len(group),
            }
        )
    return np.stack(features), pd.DataFrame(rows)


def verify_audits(audit_dirs: dict[float, Path]) -> None:
    reference: dict | None = None
    for window, directory in audit_dirs.items():
        required = (".complete", "extraction_config.json")
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete {window:g}s audit in {directory}: {', '.join(missing)}"
            )
        config = json.loads((directory / "extraction_config.json").read_text())
        if not np.isclose(float(config["window_seconds"]), window):
            raise ValueError(f"Unexpected window size in {directory}")
        if reference is None:
            reference = config
            continue
        for field in ("train", "validation", "eligibility_window_seconds"):
            if config[field] != reference[field]:
                raise ValueError(f"Audit mismatch for {field} at {window:g}s")


def verify_same_cohort(frames: dict[float, pd.DataFrame]) -> None:
    reference = frames[20.0].sort_values("recording_index").reset_index(drop=True)
    columns = ["recording_index", "audio_path", "subject_id", "class_label"]
    for window, frame in frames.items():
        candidate = frame.sort_values("recording_index").reset_index(drop=True)
        if not candidate[columns].equals(reference[columns]):
            raise ValueError(f"Recording cohort or labels differ at {window:g}s")


def plot_split(
    *,
    split: str,
    audit_dirs: dict[float, Path],
    output_dir: Path,
    neighbors: int,
    min_dist: float,
    seed: int,
) -> None:
    features: dict[float, np.ndarray] = {}
    metadata: dict[float, pd.DataFrame] = {}
    for window, directory in audit_dirs.items():
        features[window], metadata[window] = aggregate_recordings(directory, split)
    verify_same_cohort(metadata)

    coordinate_frames = []
    silhouette_rows = []
    for window in WINDOWS:
        count = len(metadata[window])
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(neighbors, count - 1),
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
            transform_seed=seed,
        )
        coordinates = reducer.fit_transform(features[window])
        frame = metadata[window].copy()
        frame.insert(0, "window_seconds", window)
        frame["umap_1"] = coordinates[:, 0]
        frame["umap_2"] = coordinates[:, 1]
        coordinate_frames.append(frame)
        labels = frame["class_label"].to_numpy(dtype=int)
        silhouette_rows.append(
            {
                "split": split,
                "window_seconds": window,
                "recordings": count,
                "original_cosine_silhouette": silhouette_score(
                    features[window], labels, metric="cosine"
                ),
                "umap_silhouette": silhouette_score(
                    frame[["umap_1", "umap_2"]], labels, metric="euclidean"
                ),
            }
        )
    coordinates_frame = pd.concat(coordinate_frames, ignore_index=True)
    coordinates_frame.to_csv(
        output_dir / f"recording_umap_coordinates_{split}.csv", index=False
    )
    pd.DataFrame(silhouette_rows).to_csv(
        output_dir / f"recording_umap_silhouette_{split}.csv", index=False
    )

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    classes = sorted(coordinates_frame["class_label"].unique())
    for axis, window in zip(axes.flat, WINDOWS):
        frame = coordinates_frame[coordinates_frame["window_seconds"] == window]
        for class_label in classes:
            selected = frame[frame["class_label"] == class_label]
            axis.scatter(
                selected["umap_1"],
                selected["umap_2"],
                s=34,
                alpha=0.78,
                color=colors(int(class_label)),
                label=CLASS_NAMES.get(int(class_label), f"Class {class_label}"),
                edgecolors="none",
            )
        axis.set_title(f"{window:g}-second windows (n={len(frame)})")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.grid(alpha=0.15)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(classes))
    figure.suptitle(
        f"Frozen RDINO recording embeddings — {split}\n"
        "Independent cosine UMAP per window size; color was not used during fitting",
        fontsize=15,
    )
    output_path = output_dir / f"recording_umap_by_class_{split}.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    if args.neighbors < 2:
        raise ValueError("--neighbors must be at least 2")
    if not 0 <= args.min_dist <= 1:
        raise ValueError("--min-dist must be between 0 and 1")
    audit_root = args.audit_root.expanduser().resolve()
    reference_20_dir = args.reference_20_dir.expanduser().resolve()
    audit_dirs = directories(audit_root, reference_20_dir)
    verify_audits(audit_dirs)
    audit_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        plot_split(
            split=split,
            audit_dirs=audit_dirs,
            output_dir=audit_root,
            neighbors=args.neighbors,
            min_dist=args.min_dist,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
