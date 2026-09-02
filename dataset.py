from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"audio_path", "word_timestamps_path", "class_label", "regression_label"}


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class Recording:
    audio_path: Path
    word_timestamps_path: Path
    class_label: int
    regression_label: float
    subject_id: str


@dataclass(frozen=True)
class Window:
    recording_index: int
    start_seconds: float
    end_seconds: float
    text: str


def _resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def read_manifest(path: str | Path, num_classes: int = 2) -> list[Recording]:
    """Read recording metadata; MultimodalDataset expands it into windows."""
    manifest_path = Path(path).expanduser().resolve()
    frame = pd.read_csv(manifest_path, dtype={"subject_id": str})
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")

    recordings: list[Recording] = []
    for row_number, row in frame.iterrows():
        location = f"{manifest_path}, CSV row {row_number + 2}"
        audio_path = _resolve_path(str(row["audio_path"]), manifest_path.parent)
        timestamps_path = _resolve_path(str(row["word_timestamps_path"]), manifest_path.parent)
        try:
            class_label = int(row["class_label"])
            regression_label = float(row["regression_label"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid label at {location}") from error
        if not 0 <= class_label < num_classes:
            raise ValueError(
                f"class_label at {location} is {class_label}; expected 0..{num_classes - 1}"
            )
        if not math.isfinite(regression_label):
            raise ValueError(f"Non-finite regression_label at {location}")
        subject = row.get("subject_id", "")
        subject_id = "" if pd.isna(subject) else str(subject)
        recordings.append(
            Recording(audio_path, timestamps_path, class_label, regression_label, subject_id)
        )
    if not recordings:
        raise ValueError(f"Manifest contains no recordings: {manifest_path}")
    return recordings


def _json_word_entries(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("words"), list):
        return payload["words"]
    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        return [word for segment in payload["segments"] for word in segment.get("words", [])]
    raise ValueError("JSON must be a word list, {'words': [...]}, or {'segments': [...]}")


def _pick(entry: dict[str, Any], names: tuple[str, ...], location: str) -> Any:
    for name in names:
        if name in entry and not pd.isna(entry[name]):
            return entry[name]
    raise ValueError(f"Missing one of {names} in {location}")


def read_word_timestamps(path: str | Path) -> list[TimedWord]:
    """Read Whisper-style JSON or CSV/TSV word timestamps."""
    timestamp_path = Path(path)
    suffix = timestamp_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(timestamp_path.read_text(encoding="utf-8"))
        entries = list(_json_word_entries(payload))
    elif suffix in {".csv", ".tsv"}:
        entries = pd.read_csv(timestamp_path, sep="\t" if suffix == ".tsv" else ",").to_dict(
            "records"
        )
    else:
        raise ValueError(
            f"Unsupported timestamp format {suffix!r} for {timestamp_path}; use JSON, CSV, or TSV"
        )

    words: list[TimedWord] = []
    for index, entry in enumerate(entries):
        location = f"{timestamp_path}, word {index + 1}"
        if not isinstance(entry, dict):
            raise ValueError(f"Expected an object at {location}")
        text = str(_pick(entry, ("word", "text", "token"), location)).strip()
        start = float(_pick(entry, ("start", "start_time", "start_seconds"), location))
        end = float(_pick(entry, ("end", "end_time", "end_seconds"), location))
        if not text:
            continue
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError(f"Invalid timestamp interval at {location}: {start}..{end}")
        words.append(TimedWord(text, start, end))
    words.sort(key=lambda word: (word.start_seconds, word.end_seconds))
    if not words:
        raise ValueError(f"No timestamped words found in {timestamp_path}")
    return words


def _window_starts(duration: float, window: float, stride: float) -> list[float]:
    if duration < window:
        return []
    if duration == window:
        return [0.0]
    starts: list[float] = []
    start = 0.0
    while start + window <= duration + 1e-8:
        starts.append(start)
        start += stride
    tail_start = duration - window
    if not starts or tail_start - starts[-1] > 1e-6:
        starts.append(tail_start)
    return starts


def _words_in_window(words: list[TimedWord], start: float, end: float) -> str:
    # Boundary words intentionally occur in both overlapping examples.
    return " ".join(
        word.text for word in words
        if word.end_seconds > start and word.start_seconds < end
    )


def _repeat_to_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
    sample_count = waveform.shape[-1]
    if sample_count == 0:
        raise ValueError("Audio window has zero samples")
    if sample_count >= target_samples:
        return waveform[:target_samples]
    return waveform.repeat(math.ceil(target_samples / sample_count))[:target_samples]


class MultimodalDataset(Dataset[dict[str, Any]]):
    """Expand each manifest recording into aligned sliding-window examples."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        sample_rate: int = 16_000,
        window_seconds: float = 55.0,
        stride_seconds: float = 10.0,
        num_classes: int = 2,
        include_empty_text: bool = False,
    ) -> None:
        if window_seconds <= 0 or stride_seconds <= 0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        self.recordings = read_manifest(manifest_path, num_classes=num_classes)
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.target_samples = round(sample_rate * window_seconds)
        self.windows: list[Window] = []
        self.skipped_short_recordings = 0

        for recording_index, recording in enumerate(self.recordings):
            if not recording.audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {recording.audio_path}")
            if not recording.word_timestamps_path.is_file():
                raise FileNotFoundError(f"Word timestamps not found: {recording.word_timestamps_path}")
            metadata = torchaudio.info(recording.audio_path)
            duration = metadata.num_frames / metadata.sample_rate
            if duration < window_seconds:
                self.skipped_short_recordings += 1
                continue
            words = read_word_timestamps(recording.word_timestamps_path)
            for start in _window_starts(duration, window_seconds, stride_seconds):
                end = min(start + window_seconds, duration)
                text = _words_in_window(words, start, end)
                if text or include_empty_text:
                    self.windows.append(Window(recording_index, start, end, text))
        if not self.windows:
            raise ValueError("No windows were produced; check audio durations and word timestamps")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        recording = self.recordings[window.recording_index]
        metadata = torchaudio.info(recording.audio_path)
        frame_offset = round(window.start_seconds * metadata.sample_rate)
        frames = round(self.window_seconds * metadata.sample_rate)
        waveform, source_rate = torchaudio.load(
            recording.audio_path, frame_offset=frame_offset, num_frames=frames
        )
        waveform = waveform.float().mean(dim=0)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        waveform = _repeat_to_length(waveform, self.target_samples)
        return {
            "waveform": waveform,
            "text": window.text,
            "class_label": recording.class_label,
            "regression_label": recording.regression_label,
            "audio_path": str(recording.audio_path),
            "subject_id": recording.subject_id,
            "window_start": window.start_seconds,
            "window_end": window.end_seconds,
        }


class MultimodalCollator:
    def __init__(self, tokenizer: Any, max_text_tokens: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        tokens = self.tokenizer(
            [item["text"] for item in examples], padding=True, truncation=True,
            max_length=self.max_text_tokens, return_tensors="pt"
        )
        return {
            "tokens": tokens,
            "waveforms": torch.stack([item["waveform"] for item in examples]),
            "class_labels": torch.tensor(
                [item["class_label"] for item in examples], dtype=torch.long
            ),
            "regression_labels": torch.tensor(
                [item["regression_label"] for item in examples], dtype=torch.float32
            ),
            "audio_paths": [item["audio_path"] for item in examples],
            "subject_ids": [item["subject_id"] for item in examples],
            "window_starts": [item["window_start"] for item in examples],
            "window_ends": [item["window_end"] for item in examples],
        }
