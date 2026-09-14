"""Create an immutable, CPU-only evaluation bundle from the full source CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf

from clinical_evaluation import (
    assign_words, block_windows, loso_folds, mixed_subject_folds,
    regression_metrics, speaker_blocks, stable_id,
)
from dataset import read_diarization, read_word_timestamps


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, values):
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, allow_nan=False) + "\n")


def prepare_recording(row, source_dir, csv_row, max_seconds, overrides, max_gap_seconds=60.0):
    for column in ("subject_id", "study"):
        if not row.get(column, "").strip():
            raise ValueError(f"Missing {column}")
    y = float(row["regression_label"])
    if not math.isfinite(y) or not 0 <= y <= 60:
        raise ValueError("Concurrent MADRS must be finite and between 0 and 60")
    paths = {}
    for column in ("audio_path", "word_timestamps_path", "diarization_path"):
        value = row.get(column, "").strip()
        if not value:
            raise ValueError(f"Missing {column}")
        path = Path(value).expanduser()
        path = (source_dir / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise ValueError(f"Missing file for {column}: {path}")
        paths[column] = path
    metadata = sf.info(paths["audio_path"])
    diarization = read_diarization(paths["diarization_path"])
    timed_words = read_word_timestamps(paths["word_timestamps_path"])
    words = [{"word": w.text, "start": w.start_seconds, "end": w.end_seconds} for w in timed_words]
    recording_id = stable_id(str(paths["audio_path"]))
    primary, intervals, qc = speaker_blocks(
        [(s.start_seconds, s.end_seconds, s.speaker) for s in diarization],
        metadata.duration, overrides.get(recording_id), max_gap_seconds,
    )
    if not intervals:
        raise ValueError("No primary-speaker audio after removing other speakers")
    assigned = assign_words(words, intervals)
    flags = []
    if qc["speaker_count"] != 2:
        flags.append("speaker_count_not_two")
    if qc["primary_speech_share"] < .6:
        flags.append("primary_speech_share_below_0.6")
    if qc["primary_duration_tie"]:
        flags.append("primary_duration_tie")
    if qc["excluded_long_gaps"]:
        flags.append("long_gap_cropped")
    if qc["diarization_segments_clipped"]:
        flags.append("diarization_clipped_to_audio")
    if qc["removed_overlap_seconds"] > 1e-8:
        flags.append("overlapping_other_speech_removed")
    if any(w["end"] > metadata.duration for w in words):
        flags.append("word_timestamps_past_audio")
    if max(b - a for a, b in intervals) < max_seconds:
        flags.append("all_blocks_shorter_than_window")
    retained_words = sum(map(len, assigned))
    if not retained_words:
        raise ValueError("No unambiguous subject words; common multimodal cohort would be empty for this recording")
    record = {
        "recording_id": recording_id,
        **{k: str(v) for k, v in paths.items()},
        "class_label": row.get("class_label", ""),
        "regression_label": y, "subject_id": row["subject_id"].strip(),
        "study": row["study"].strip(), "source_csv_row": csv_row,
        "primary_speaker": primary, "audio_duration_seconds": metadata.duration,
        "sample_rate": metadata.samplerate, "channels": metadata.channels,
        "block_count": len(intervals), "retained_word_count": retained_words,
        "retained_block_seconds": qc["retained_block_seconds"],
        "qc_flags": "|".join(flags),
    }
    blocks, windows = [], []
    for index, ((start, end), block_words) in enumerate(zip(intervals, assigned)):
        block_id = f"{recording_id}:b{index:04d}"
        blocks.append({
            "recording_id": recording_id, "block_id": block_id,
            "start_seconds": start, "end_seconds": end,
            "words": block_words, "text": " ".join(w["word"] for w in block_words),
        })
        for window_index, (left, right, window_words) in enumerate(block_windows(start, end, block_words, max_seconds)):
            windows.append({
                "recording_id": recording_id, "block_id": block_id,
                "window_id": f"{block_id}:w{window_index:04d}",
                "audio_path": str(paths["audio_path"]),
                "start_seconds": left, "end_seconds": right,
                "duration_seconds": right - left,
                "aggregation_weight": (right - left) / qc["retained_block_seconds"],
                "word_count": len(window_words),
                "text": " ".join(w["word"] for w in window_words),
            })
    qc.update({
        "recording_id": recording_id, "source_csv_row": csv_row,
        "study": record["study"], "primary_speaker": primary,
        "speaker_override_used": recording_id in overrides,
        "original_word_count": len(words), "retained_word_count": retained_words,
        "omitted_word_count": len(words) - retained_words,
        "flags": flags,
        "word_timestamps_sha256": digest(paths["word_timestamps_path"]),
        "diarization_sha256": digest(paths["diarization_path"]),
        "audio_size_bytes": paths["audio_path"].stat().st_size,
        "audio_mtime_ns": paths["audio_path"].stat().st_mtime_ns,
    })
    return record, blocks, windows, qc


def build(args):
    source = args.source.expanduser().resolve()
    if args.output.exists():
        raise ValueError("Output already exists; use a NEW version directory")
    overrides = json.loads(args.speaker_overrides.read_text()) if args.speaker_overrides else {}
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty input CSV")
    records, blocks, windows, audits, errors = [], [], [], [], []
    seen = set()
    for index, row in enumerate(rows):
        try:
            record, new_blocks, new_windows, qc = prepare_recording(
                row, source.parent, index + 2, args.window_seconds, overrides, args.max_gap_seconds,
            )
            if record["recording_id"] in seen:
                raise ValueError("Duplicate resolved audio path in CSV")
            seen.add(record["recording_id"])
            records.append(record)
            blocks.extend(new_blocks)
            windows.extend(new_windows)
            audits.append(qc)
        except Exception as error:
            errors.append({"source_csv_row": index + 2, "error": str(error)})
        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(rows)} recordings", flush=True)
    if errors:
        print(json.dumps({"validation_errors": errors}, indent=2))
        raise ValueError("Bundle not written: fix source problems; no silent exclusions")
    unknown = set(overrides) - seen
    if unknown:
        raise ValueError(f"Unknown recording IDs in speaker overrides: {sorted(unknown)}")
    folds = loso_folds(records)
    mixed = mixed_subject_folds(records, seed=args.seed)
    args.output.mkdir(parents=True)
    write_csv(args.output / "recordings.csv", records)
    write_jsonl(args.output / "blocks.jsonl", blocks)
    write_csv(args.output / "windows.csv", windows)
    write_jsonl(args.output / "recording_audit.jsonl", audits)
    reviews = [{k: q[k] for k in ("recording_id", "source_csv_row", "study", "primary_speaker", "primary_speech_share", "max_preserved_gap_seconds")} | {"flags": "|".join(q["flags"])} for q in audits if q["flags"]]
    write_csv(args.output / "review_queue.csv", reviews, ["recording_id", "source_csv_row", "study", "primary_speaker", "primary_speech_share", "max_preserved_gap_seconds", "flags"])
    split_summary = []
    for fold in folds:
        directory = args.output / "loso" / fold["test_study"]
        write_csv(directory / "train.csv", fold["train"])
        write_csv(directory / "test.csv", fold["test"])
        for inner in fold["inner"]:
            inner_dir = directory / "inner" / inner["validation_study"]
            write_csv(inner_dir / "train.csv", inner["train"])
            write_csv(inner_dir / "validation.csv", inner["validation"])
        split_summary.append({
            "test_study": fold["test_study"],
            "train_recordings": len(fold["train"]), "test_recordings": len(fold["test"]),
            "train_subjects": len({r["subject_id"] for r in fold["train"]}),
            "test_subjects": len({r["subject_id"] for r in fold["test"]}),
            "purged_recordings": len(records) - len(fold["train"]) - len(fold["test"]),
            "training_mean_madrs": sum(r["regression_label"] for r in fold["train"]) / len(fold["train"]),
        })
    for fold in mixed:
        directory = args.output / "mixed_subject" / f"fold_{fold['fold']}"
        write_csv(directory / "train.csv", fold["train"])
        write_csv(directory / "test.csv", fold["test"])
        # Reuse the global held-out subject groups for four inner validation folds.
        for candidate in mixed:
            if candidate["fold"] == fold["fold"]:
                continue
            val_subjects = {r["subject_id"] for r in candidate["test"]}
            inner_dir = directory / "inner" / f"fold_{candidate['fold']}"
            write_csv(inner_dir / "train.csv", [r for r in fold["train"] if r["subject_id"] not in val_subjects])
            write_csv(inner_dir / "validation.csv", [r for r in fold["train"] if r["subject_id"] in val_subjects])
    write_csv(args.output / "split_summary.csv", split_summary)
    study_summary = []
    for study in sorted({r["study"] for r in records}):
        subset = [r for r in records if r["study"] == study]
        labels = [r["regression_label"] for r in subset]
        metrics = regression_metrics(labels, labels)
        study_summary.append({"study": study, "subjects": len({r["subject_id"] for r in subset}), **{k: metrics[k] for k in ("recordings", "madrs_mean", "madrs_sd", "madrs_min", "madrs_max")}})
    write_csv(args.output / "study_summary.csv", study_summary)
    summary = {
        "recordings": len(records), "subjects": len({r["subject_id"] for r in records}),
        "studies": len(study_summary), "blocks": len(blocks), "windows": len(windows),
        "short_windows": sum(w["duration_seconds"] < args.window_seconds - 1e-8 for w in windows),
        "empty_text_windows": sum(not w["text"] for w in windows),
        "excluded_long_gap_count": sum(len(q["excluded_long_gaps"]) for q in audits),
        "excluded_long_gap_seconds": sum(q["excluded_long_gap_seconds"] for q in audits),
        "review_recordings": len(reviews),
        "qc_flag_counts": dict(Counter(flag for q in audits for flag in q["flags"])),
    }
    config = {
        "schema_version": 2, "source_csv": str(source), "source_csv_sha256": digest(source),
        "window_seconds": args.window_seconds, "mixed_subject_seed": args.seed,
        "speaker_selection": "unioned in-bounds duration, before filling gaps",
        "speaker_overrides": overrides,
        "overlap_policy": "subtract every non-primary speaker interval",
        "gap_policy": "preserve same-primary-speaker gaps up to max_gap_seconds; crop longer gaps into separate blocks",
        "max_gap_seconds": args.max_gap_seconds,
        "word_policy": "full containment within subject block; no ambiguous boundary words",
        "window_policy": "nonoverlapping, word-aware boundaries, retain short blocks and tails",
        "aggregation": "duration-weighted window predictions, then one prediction per recording",
        "empty_text_policy": "all view retains every window; text view uses word_count > 0 and renormalizes duration weights within each recording",
        "audio_integrity": "file size and mtime recorded; audio content is not hashed",
        "class_label": "copied unchanged for compatibility, unused for splits or scoring",
        "summary": summary,
        "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): digest(p) for p in [Path(__file__).resolve(), Path(__file__).resolve().parents[1] / "clinical_evaluation.py", Path(__file__).resolve().parents[1] / "dataset.py"]},
    }
    write_json(args.output / "config.json", config)
    # Content checksums make later modifications detectable, including transcript text.
    files = sorted(p for p in args.output.rglob("*") if p.is_file())
    write_json(args.output / "checksums.json", {str(p.relative_to(args.output)): digest(p) for p in files})
    print(json.dumps(summary, indent=2))
    print(f"Bundle written to {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=55.0)
    parser.add_argument("--max-gap-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--speaker-overrides", type=Path, help="JSON mapping recording_id to diarization speaker label")
    args = parser.parse_args()
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error("--window-seconds must be finite and positive")
    if not math.isfinite(args.max_gap_seconds) or args.max_gap_seconds < 0:
        parser.error("--max-gap-seconds must be finite and nonnegative")
    build(args)


if __name__ == "__main__":
    main()
