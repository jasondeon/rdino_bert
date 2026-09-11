from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MultimodalDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export exact model-input audio windows as listenable WAV files"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/audio-window-samples")
    )
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--stride-seconds", type=float, default=15.0)
    parser.add_argument("--eligibility-window-seconds", type=float, default=30.0)
    parser.add_argument(
        "--speaker-gap-policy",
        choices=("preserve", "concatenate"),
        default="preserve",
    )
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--include-empty-text", action="store_true")
    parser.add_argument(
        "--selection",
        choices=("mixed", "random", "first"),
        default="mixed",
        help=(
            "Mixed prioritizes both multi-span cut windows and single-span windows, "
            "with at most one sample per recording when possible (default: mixed)"
        ),
    )
    parser.add_argument(
        "--window-index",
        action="append",
        type=int,
        help="Export this exact dataset window index; may be repeated",
    )
    parser.add_argument(
        "--subject-id",
        action="append",
        help="Restrict automatic selection to this subject ID; may be repeated",
    )
    parser.add_argument(
        "--no-source-context",
        action="store_true",
        help="Do not export the continuous original audio between the first and last span",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return cleaned or "unknown"


def choose_indices(dataset: MultimodalDataset, args: argparse.Namespace) -> list[int]:
    if args.window_index:
        unique = list(dict.fromkeys(args.window_index))
        invalid = [index for index in unique if not 0 <= index < len(dataset)]
        if invalid:
            raise IndexError(f"Window indices out of range: {invalid}")
        return unique

    allowed_subjects = set(args.subject_id or [])
    candidates = [
        index
        for index, window in enumerate(dataset.windows)
        if not allowed_subjects
        or dataset.recordings[window.recording_index].subject_id in allowed_subjects
    ]
    if not candidates:
        raise ValueError("No windows match the requested subject restriction")
    rng = np.random.default_rng(args.seed)
    if args.selection == "first":
        return candidates[: args.count]

    def distinct_recordings(pool: list[int], limit: int, used: set[int]) -> list[int]:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        selected = []
        for index in shuffled:
            recording_index = dataset.windows[index].recording_index
            if recording_index in used:
                continue
            selected.append(index)
            used.add(recording_index)
            if len(selected) == limit:
                break
        return selected

    used_recordings: set[int] = set()
    selected: list[int] = []
    if args.selection == "mixed":
        multi_span = [
            index for index in candidates if len(dataset.windows[index].source_spans) > 1
        ]
        single_span = [
            index for index in candidates if len(dataset.windows[index].source_spans) == 1
        ]
        multi_target = min((args.count + 1) // 2, len(multi_span))
        selected.extend(distinct_recordings(multi_span, multi_target, used_recordings))
        selected.extend(
            distinct_recordings(
                single_span, args.count - len(selected), used_recordings
            )
        )
    else:
        selected.extend(distinct_recordings(candidates, args.count, used_recordings))

    if len(selected) < args.count:
        remaining = [index for index in candidates if index not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: args.count - len(selected)])
    rng.shuffle(selected)
    return selected[: args.count]


def read_continuous_context(
    audio_path: Path, start_seconds: float, end_seconds: float, sample_rate: int
) -> torch.Tensor:
    metadata = sf.info(audio_path)
    frame_start = max(0, math.floor(start_seconds * metadata.samplerate))
    frame_end = min(metadata.frames, math.ceil(end_seconds * metadata.samplerate))
    samples, source_rate = sf.read(
        audio_path,
        start=frame_start,
        frames=frame_end - frame_start,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(samples).mean(dim=1)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform


def audio_statistics(waveform: torch.Tensor) -> dict[str, float]:
    return {
        "waveform_mean": float(waveform.mean()),
        "waveform_rms": float(waveform.square().mean().sqrt()),
        "waveform_peak": float(waveform.abs().max()),
    }


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    manifest = resolve_path(args.manifest)
    output_dir = resolve_path(args.output_dir)
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

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
    selected = choose_indices(dataset, args)
    rows: list[dict[str, Any]] = []
    for sample_number, window_index in enumerate(selected, start=1):
        item = dataset[window_index]
        window = dataset.windows[window_index]
        recording = dataset.recordings[window.recording_index]
        waveform = item["waveform"].detach().cpu().float()
        stem = (
            f"{sample_number:02d}_window-{window_index:05d}_"
            f"subject-{safe_name(recording.subject_id)}"
        )
        model_path = output_dir / f"{stem}_model-input.wav"
        sf.write(model_path, waveform.numpy(), dataset.sample_rate, subtype="FLOAT")

        context_path: Path | None = None
        if not args.no_source_context:
            context = read_continuous_context(
                recording.audio_path,
                window.source_spans[0].start_seconds,
                window.source_spans[-1].end_seconds,
                dataset.sample_rate,
            )
            context_path = output_dir / f"{stem}_source-context.wav"
            sf.write(context_path, context.numpy(), dataset.sample_rate, subtype="FLOAT")

        spans = [
            {
                "start_seconds": span.start_seconds,
                "end_seconds": span.end_seconds,
            }
            for span in window.source_spans
        ]
        source_speech_seconds = sum(
            span.end_seconds - span.start_seconds for span in window.source_spans
        )
        wallclock_seconds = (
            window.source_spans[-1].end_seconds
            - window.source_spans[0].start_seconds
        )
        rows.append(
            {
                "sample_number": sample_number,
                "window_index": window_index,
                "recording_index": window.recording_index,
                "subject_id": recording.subject_id,
                "model_input_wav": model_path.name,
                "source_context_wav": context_path.name if context_path else "",
                "original_audio_path": str(recording.audio_path),
                "word_timestamps_path": str(recording.word_timestamps_path),
                "diarization_path": str(recording.diarization_path),
                "speaker": window.speaker,
                "class_label": recording.class_label,
                "regression_label": recording.regression_label,
                "transcript": item["text"],
                "source_spans": json.dumps(spans),
                "source_span_count": len(spans),
                "source_speech_seconds": source_speech_seconds,
                "wallclock_context_seconds": wallclock_seconds,
                "removed_gap_seconds": wallclock_seconds - source_speech_seconds,
                "model_input_samples": waveform.numel(),
                "model_input_seconds": waveform.numel() / dataset.sample_rate,
                "sample_rate": dataset.sample_rate,
                **audio_statistics(waveform),
            }
        )
        print(
            f"Wrote {model_path.name}: {len(spans)} source span(s), "
            f"{wallclock_seconds - source_speech_seconds:.3f}s removed gaps",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(output_dir / "samples.csv", index=False)
    configuration = {
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "selection": args.selection,
        "seed": args.seed,
        "requested_count": args.count,
        "exported_count": len(rows),
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "eligibility_window_seconds": args.eligibility_window_seconds,
        "sample_rate": dataset.sample_rate,
        "speaker_gap_policy": args.speaker_gap_policy,
        "wav_subtype": "32-bit IEEE float",
        "augmentation": "none",
        "stride_jitter_seconds": 0.0,
    }
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        "Each *_model-input.wav is the exact mono 16 kHz waveform returned by "
        "MultimodalDataset and passed to RDINO before MelSpectrogram conversion.\n"
        "Each *_source-context.wav preserves the continuous original interval from "
        "the first retained source span to the last, making removed diarization gaps "
        "audible by comparison. No augmentation or stride jitter was applied.\n"
        "See samples.csv for transcripts, labels, source spans, and signal levels.\n",
        encoding="utf-8",
    )
    print(f"Metadata: {output_dir / 'samples.csv'}")


if __name__ == "__main__":
    main()
