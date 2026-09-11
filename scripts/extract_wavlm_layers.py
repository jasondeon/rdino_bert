from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, WavLMModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MultimodalDataset


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
        description="Extract temporal statistics from every frozen WavLM layer"
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--stride-seconds", type=float, default=7.5)
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
        "--cache-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cpu, cuda, or cuda:0 (default: auto)",
    )
    parser.add_argument("--dry-run", action="store_true")
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


def masked_statistics(
    hidden_state: torch.Tensor, feature_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    values = hidden_state.float()
    mask = feature_mask.unsqueeze(-1).to(values.dtype)
    counts = mask.sum(dim=1).clamp_min(1.0)
    means = (values * mask).sum(dim=1) / counts
    variances = ((values - means.unsqueeze(1)).square() * mask).sum(dim=1) / counts
    return means, variances.clamp_min(0.0).sqrt()


def extract_split(
    *,
    split: str,
    manifest: Path,
    output_dir: Path,
    model: WavLMModel,
    feature_extractor: AutoFeatureExtractor,
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
    layer_count = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    cache_dtype = np.dtype(args.cache_dtype)
    means_cache = open_memmap(
        output_dir / f"{split}_layer_means.npy",
        mode="w+",
        dtype=cache_dtype,
        shape=(len(dataset), layer_count, hidden_size),
    )
    stds_cache = open_memmap(
        output_dir / f"{split}_layer_stds.npy",
        mode="w+",
        dtype=cache_dtype,
        shape=(len(dataset), layer_count, hidden_size),
    )
    rows: list[dict[str, Any]] = []
    written = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            processed = feature_extractor(
                [waveform.numpy() for waveform in batch["waveforms"]],
                sampling_rate=16_000,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_values = processed.input_values.to(device, non_blocking=True)
            attention_mask = processed.attention_mask.to(device, non_blocking=True)
            outputs = model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None or len(hidden_states) != layer_count:
                raise RuntimeError(
                    f"Expected {layer_count} hidden states, received "
                    f"{0 if hidden_states is None else len(hidden_states)}"
                )
            feature_mask = model._get_feature_vector_attention_mask(
                hidden_states[0].shape[1], attention_mask
            )
            means = []
            stds = []
            for hidden_state in hidden_states:
                layer_mean, layer_std = masked_statistics(hidden_state, feature_mask)
                means.append(layer_mean)
                stds.append(layer_std)
            means_array = torch.stack(means, dim=1).cpu().numpy()
            stds_array = torch.stack(stds, dim=1).cpu().numpy()
            count = len(batch["recording_indices"])
            means_cache[written : written + count] = means_array.astype(
                cache_dtype, copy=False
            )
            stds_cache[written : written + count] = stds_array.astype(
                cache_dtype, copy=False
            )
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
                    f"split={split} batch={batch_number}/{len(loader)} "
                    f"windows={written}",
                    flush=True,
                )
    means_cache.flush()
    stds_cache.flush()
    if written != len(dataset):
        raise RuntimeError(f"Wrote {written} embeddings for {len(dataset)} windows")
    pd.DataFrame(rows).to_csv(output_dir / f"{split}_windows.csv", index=False)
    result = {
        "manifest": str(manifest),
        "recordings": len({row["recording_index"] for row in rows}),
        "windows": written,
        "layers": layer_count,
        "hidden_size": hidden_size,
        "skipped_short_recordings": dataset.skipped_short_recordings,
    }
    print(f"Completed {split}: {json.dumps(result)}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    paths = {
        "train": resolve_path(args.train_manifest),
        "validation": resolve_path(args.validation_manifest),
        "output_dir": resolve_path(args.output_dir),
    }
    for split in ("train", "validation"):
        if not paths[split].is_file():
            raise FileNotFoundError(f"{split} manifest not found: {paths[split]}")
    if args.window_seconds <= 0 or args.stride_seconds <= 0:
        raise ValueError("Window and stride must be positive")
    if args.eligibility_window_seconds <= 0:
        raise ValueError("Eligibility window must be positive")
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Batch size must be positive and workers cannot be negative")
    device = resolve_device(args.device)
    configuration: dict[str, Any] = {
        "train": str(paths["train"]),
        "validation": str(paths["validation"]),
        "output_dir": str(paths["output_dir"]),
        "model_name": args.model_name,
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "eligibility_window_seconds": args.eligibility_window_seconds,
        "speaker_gap_policy": args.speaker_gap_policy,
        "include_empty_text": args.include_empty_text,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "cache_dtype": args.cache_dtype,
        "device": str(device),
    }
    print(json.dumps(configuration, indent=2))
    if args.dry_run:
        print("Dry run complete; model loading and extraction were not launched.")
        return

    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        output_dir / f"{split}_{suffix}"
        for split in ("train", "validation")
        for suffix in ("layer_means.npy", "layer_stds.npy", "windows.csv")
    ]
    existing = [path for path in expected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to mix or overwrite cached outputs: "
            + ", ".join(str(path) for path in existing)
        )

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = WavLMModel.from_pretrained(args.model_name).to(device)
    model.eval()
    configuration["layers"] = int(model.config.num_hidden_layers) + 1
    configuration["hidden_size"] = int(model.config.hidden_size)
    configuration["layer_names"] = ["feature_projection"] + [
        f"transformer_{index:02d}"
        for index in range(1, int(model.config.num_hidden_layers) + 1)
    ]
    split_results = {}
    for split in ("train", "validation"):
        split_results[split] = extract_split(
            split=split,
            manifest=paths[split],
            output_dir=output_dir,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            args=args,
        )
    configuration["splits"] = split_results
    (output_dir / "extraction_config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
