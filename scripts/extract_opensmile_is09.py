from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import opensmile
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MultimodalDataset


_THREAD_STATE = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract official openSMILE IS09 functionals from audio windows"
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=7.5)
    parser.add_argument("--eligibility-window-seconds", type=float, default=30.0)
    parser.add_argument(
        "--speaker-gap-policy", choices=("preserve", "concatenate"), default="preserve"
    )
    parser.add_argument(
        "--window-selection",
        choices=("all", "first_per_recording"),
        default="all",
        help=(
            "Extract every eligible window, or only the earliest window from the "
            "earliest primary-speaker run that supports the requested duration"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audio-workers", type=int, default=2)
    parser.add_argument("--smile-workers", type=int, default=8)
    parser.add_argument("--include-empty-text", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class AudioCollator:
    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "waveforms": [item["waveform"].numpy() for item in examples],
            "recording_indices": [item["recording_index"] for item in examples],
            "audio_paths": [item["audio_path"] for item in examples],
            "subject_ids": [item["subject_id"] for item in examples],
            "speakers": [item["speaker"] for item in examples],
            "window_starts": [item["window_start"] for item in examples],
            "window_ends": [item["window_end"] for item in examples],
            "class_labels": [item["class_label"] for item in examples],
            "regression_labels": [item["regression_label"] for item in examples],
        }


def thread_smile() -> opensmile.Smile:
    if not hasattr(_THREAD_STATE, "smile"):
        _THREAD_STATE.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.IS09,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _THREAD_STATE.smile


def extract_signal(waveform: np.ndarray) -> np.ndarray:
    result = thread_smile().process_signal(waveform[np.newaxis, :], 16_000)
    values = result.to_numpy(dtype=np.float32)
    if values.shape != (1, 384):
        raise RuntimeError(f"Expected one row of 384 IS09 features, received {values.shape}")
    return values[0]


def extract_split(
    split: str,
    manifest: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset = MultimodalDataset(
        manifest,
        sample_rate=16_000,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        num_classes=4,
        include_empty_text=args.include_empty_text,
        load_audio=True,
        eligibility_window_seconds=args.eligibility_window_seconds,
        speaker_gap_policy=args.speaker_gap_policy,
    )
    selected_indices = list(range(len(dataset)))
    if args.window_selection == "first_per_recording":
        seen_recordings: set[int] = set()
        selected_indices = []
        for index, window in enumerate(dataset.windows):
            if window.recording_index not in seen_recordings:
                selected_indices.append(index)
                seen_recordings.add(window.recording_index)
    selected_dataset = Subset(dataset, selected_indices)
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.audio_workers,
        collate_fn=AudioCollator(),
        persistent_workers=args.audio_workers > 0,
    )
    cache = open_memmap(
        output_dir / f"{split}_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(selected_dataset), 384),
    )
    rows = []
    written = 0
    with ThreadPoolExecutor(max_workers=args.smile_workers) as executor:
        for batch_number, batch in enumerate(loader, start=1):
            extracted = np.stack(list(executor.map(extract_signal, batch["waveforms"])))
            count = len(extracted)
            cache[written : written + count] = extracted
            for offset in range(count):
                rows.append(
                    {
                        "embedding_index": written + offset,
                        "recording_index": batch["recording_indices"][offset],
                        "audio_path": batch["audio_paths"][offset],
                        "subject_id": batch["subject_ids"][offset],
                        "speaker": batch["speakers"][offset],
                        "window_start": batch["window_starts"][offset],
                        "window_end": batch["window_ends"][offset],
                        "class_label": batch["class_labels"][offset],
                        "regression_label": batch["regression_labels"][offset],
                    }
                )
            written += count
            if batch_number == 1 or batch_number % 50 == 0:
                print(
                    f"split={split} batch={batch_number}/{len(loader)} windows={written}",
                    flush=True,
                )
    cache.flush()
    if written != len(selected_dataset) or not np.isfinite(cache).all():
        raise RuntimeError(
            f"Invalid {split} extraction: wrote {written}/{len(selected_dataset)}"
        )
    pd.DataFrame(rows).to_csv(output_dir / f"{split}_windows.csv", index=False)
    result = {
        "manifest": str(manifest),
        "recordings": len({row["recording_index"] for row in rows}),
        "windows": written,
        "features": 384,
        "window_selection": args.window_selection,
        "skipped_short_recordings": dataset.skipped_short_recordings,
    }
    print(f"Completed {split}: {json.dumps(result)}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    if args.window_seconds <= 0 or args.stride_seconds <= 0:
        raise ValueError("Window and stride must be positive")
    if args.batch_size < 1 or args.audio_workers < 0 or args.smile_workers < 1:
        raise ValueError("Invalid worker or batch setting")
    paths = {
        "train": args.train_manifest.expanduser().resolve(),
        "validation": args.validation_manifest.expanduser().resolve(),
        "output": args.output_dir.expanduser().resolve(),
    }
    for split in ("train", "validation"):
        if not paths[split].is_file():
            raise FileNotFoundError(paths[split])
    feature_names = thread_smile().feature_names
    config = {
        "feature_set": "IS09",
        "feature_level": "Functionals",
        "opensmile_version": opensmile.__version__,
        "feature_count": len(feature_names),
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "eligibility_window_seconds": args.eligibility_window_seconds,
        "speaker_gap_policy": args.speaker_gap_policy,
        "window_selection": args.window_selection,
        "include_empty_text": args.include_empty_text,
        "batch_size": args.batch_size,
        "audio_workers": args.audio_workers,
        "smile_workers": args.smile_workers,
        "train": str(paths["train"]),
        "validation": str(paths["validation"]),
        "output_dir": str(paths["output"]),
    }
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; no audio was loaded and no features were extracted.")
        return
    paths["output"].mkdir(parents=True, exist_ok=True)
    pd.Series(feature_names, name="feature_name").to_csv(
        paths["output"] / "feature_names.csv", index=False
    )
    config["splits"] = {
        split: extract_split(split, paths[split], paths["output"], args)
        for split in ("train", "validation")
    }
    (paths["output"] / "extraction_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
