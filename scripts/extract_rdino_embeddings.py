from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MultimodalDataset
from model import BertRdinoModel


class AudioCollator:
    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "waveforms": torch.stack([item["waveform"] for item in examples]),
            "recording_indices": [item["recording_index"] for item in examples],
            "audio_paths": [item["audio_path"] for item in examples],
            "subject_ids": [item["subject_id"] for item in examples],
            "speakers": [item["speaker"] for item in examples],
            "window_starts": [item["window_start"] for item in examples],
            "window_ends": [item["window_end"] for item in examples],
            "class_labels": [item["class_label"] for item in examples],
            "regression_labels": [item["regression_label"] for item in examples],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen pretrained RDINO embeddings for an audit"
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--rdino-yaml", type=Path, default=Path("assets/rdino.yaml"))
    parser.add_argument("--rdino-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--stride-seconds", type=float, default=15.0)
    parser.add_argument("--eligibility-window-seconds", type=float, default=30.0)
    parser.add_argument(
        "--speaker-gap-policy",
        choices=("preserve", "concatenate"),
        default="preserve",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--include-empty-text", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cpu, cuda, or cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and show configuration without extracting embeddings",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {name}")
    return device


def extract_split(
    *,
    split: str,
    manifest: Path,
    output_dir: Path,
    model: BertRdinoModel,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dataset = MultimodalDataset(
        manifest,
        sample_rate=16_000,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        num_classes=args.num_classes,
        include_empty_text=args.include_empty_text,
        load_audio=True,
        eligibility_window_seconds=args.eligibility_window_seconds,
        speaker_gap_policy=args.speaker_gap_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=AudioCollator(),
    )
    embedding_batches: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            features = model.feature_extractor(waveforms)
            embeddings = model.rdino_backbone(features).float().cpu().numpy()
            embedding_batches.append(embeddings)
            first_index = len(rows)
            for offset in range(len(batch["recording_indices"])):
                rows.append(
                    {
                        "embedding_index": first_index + offset,
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
            if batch_number == 1 or batch_number % 50 == 0:
                print(
                    f"split={split} batch={batch_number}/{len(loader)} "
                    f"windows={len(rows)}",
                    flush=True,
                )

    embedding_array = np.concatenate(embedding_batches, axis=0).astype(
        np.float32, copy=False
    )
    metadata = pd.DataFrame(rows)
    if len(metadata) != len(embedding_array):
        raise RuntimeError("Embedding and metadata row counts differ")
    np.save(output_dir / f"{split}_embeddings.npy", embedding_array)
    metadata.to_csv(output_dir / f"{split}_windows.csv", index=False)
    result = {
        "manifest": str(manifest),
        "recordings": int(metadata["recording_index"].nunique()),
        "windows": len(metadata),
        "embedding_dimension": int(embedding_array.shape[1]),
        "skipped_short_recordings": dataset.skipped_short_recordings,
    }
    print(f"Completed {split}: {json.dumps(result)}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    paths = {
        "train": resolve_path(args.train_manifest),
        "validation": resolve_path(args.validation_manifest),
        "rdino_yaml": resolve_path(args.rdino_yaml),
        "rdino_checkpoint": resolve_path(args.rdino_checkpoint),
        "output_dir": resolve_path(args.output_dir),
    }
    for name in ("train", "validation", "rdino_yaml", "rdino_checkpoint"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"{name} not found: {paths[name]}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.window_seconds <= 0 or args.stride_seconds <= 0:
        raise ValueError("Window and stride must be positive")
    if args.eligibility_window_seconds <= 0:
        raise ValueError("Eligibility window must be positive")
    device = resolve_device(args.device)
    configuration = {
        **{name: str(path) for name, path in paths.items()},
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "eligibility_window_seconds": args.eligibility_window_seconds,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "device": str(device),
        "include_empty_text": args.include_empty_text,
        "speaker_gap_policy": args.speaker_gap_policy,
    }
    print(json.dumps(configuration, indent=2))
    if args.dry_run:
        print("Dry run complete; embedding extraction was not launched.")
        return

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / f"{split}_{suffix}"
        for split in ("train", "validation")
        for suffix in ("embeddings.npy", "windows.csv")
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to mix or overwrite cached audit outputs: "
            + ", ".join(str(path) for path in existing)
        )

    model = BertRdinoModel(
        text_model_name="mental/mental-bert-base-uncased",
        rdino_yaml=paths["rdino_yaml"],
        rdino_checkpoint=paths["rdino_checkpoint"],
        modality="audio",
        use_rdino_lora=False,
        embedding_normalization="none",
        fusion_architecture="single_layer",
        freeze_rdino_batchnorm_stats=True,
    ).to(device)
    model.eval()
    split_results = {}
    for split in ("train", "validation"):
        split_results[split] = extract_split(
            split=split,
            manifest=paths[split],
            output_dir=output_dir,
            model=model,
            device=device,
            args=args,
        )
    configuration["splits"] = split_results
    (output_dir / "extraction_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
