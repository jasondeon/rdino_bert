from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


CLASS_NAMES = {
    0: "Class 0 (0–4)",
    1: "Class 1 (5–11)",
    2: "Class 2 (12–22)",
    3: "Class 3 (≥23)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PCA of recording-level frozen RDINO embeddings"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/rdino-embedding-audit-preserved-gaps"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/rdino-window-size-audit/recording_pca_by_class_20s.png"
        ),
    )
    return parser.parse_args()


def aggregate_recordings(directory: Path, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(directory / f"{split}_embeddings.npy", mmap_mode="r")
    windows = pd.read_csv(
        directory / f"{split}_windows.csv", dtype={"subject_id": str}
    )
    if embeddings.ndim != 2 or len(embeddings) != len(windows):
        raise ValueError(f"Invalid {split} embedding cache in {directory}")

    features: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for recording_index, group in windows.groupby("recording_index", sort=True):
        indices = group["embedding_index"].to_numpy(dtype=int)
        labels = group["class_label"].to_numpy(dtype=int)
        if not np.all(labels == labels[0]):
            raise ValueError(f"Class labels differ within recording {recording_index}")
        embedding = np.asarray(embeddings[indices], dtype=np.float64).mean(axis=0)
        norm = np.linalg.norm(embedding)
        if norm <= 0:
            raise ValueError(f"Zero-norm embedding for recording {recording_index}")
        features.append(embedding / norm)
        first = group.iloc[0]
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


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not (input_dir / ".complete").exists():
        raise FileNotFoundError(f"Embedding audit is incomplete: {input_dir}")
    config = json.loads((input_dir / "extraction_config.json").read_text())
    window_seconds = float(config["window_seconds"])

    train_features, train_metadata = aggregate_recordings(input_dir, "train")
    validation_features, validation_metadata = aggregate_recordings(
        input_dir, "validation"
    )
    pca = PCA(n_components=2, svd_solver="full")
    train_coordinates = pca.fit_transform(train_features)
    validation_coordinates = pca.transform(validation_features)
    explained = pca.explained_variance_ratio_

    output_path.parent.mkdir(parents=True, exist_ok=True)
    coordinate_frames = []
    for split, metadata, coordinates in (
        ("train", train_metadata, train_coordinates),
        ("validation", validation_metadata, validation_coordinates),
    ):
        frame = metadata.copy()
        frame.insert(0, "split", split)
        frame["pc1"] = coordinates[:, 0]
        frame["pc2"] = coordinates[:, 1]
        coordinate_frames.append(frame)
    pd.concat(coordinate_frames, ignore_index=True).to_csv(
        output_path.with_name(f"{output_path.stem}_coordinates.csv"), index=False
    )
    variance = {
        "fit_split": "train",
        "window_seconds": window_seconds,
        "normalization": "recording mean followed by L2 normalization",
        "pc1_explained_variance_ratio": float(explained[0]),
        "pc2_explained_variance_ratio": float(explained[1]),
        "two_component_cumulative_explained_variance_ratio": float(explained.sum()),
    }
    variance_path = output_path.with_name(f"{output_path.stem}_variance.json")
    variance_path.write_text(json.dumps(variance, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    classes = sorted(train_metadata["class_label"].unique())
    for axis, split, metadata, coordinates in (
        (axes[0], "Training", train_metadata, train_coordinates),
        (axes[1], "Validation", validation_metadata, validation_coordinates),
    ):
        labels = metadata["class_label"].to_numpy(dtype=int)
        for class_label in classes:
            selected = labels == class_label
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                s=36,
                alpha=0.78,
                color=colors(int(class_label)),
                edgecolors="none",
                label=CLASS_NAMES.get(class_label, f"Class {class_label}"),
            )
        axis.set_title(f"{split} recordings (n={len(metadata)})")
        axis.set_xlabel(f"PC1 ({explained[0] * 100:.2f}%)")
        axis.set_ylabel(f"PC2 ({explained[1] * 100:.2f}%)")
        axis.grid(alpha=0.15)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(classes))
    figure.suptitle(
        f"Frozen RDINO recording embeddings — {window_seconds:g}-second windows\n"
        "PCA fit on L2-normalized training recording means",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(json.dumps(variance, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
