"""Model-independent speech boundaries, folds, and regression scoring.

Intervals are half-open seconds on the ORIGINAL recording timeline. No function
in this module joins waveforms. Subject IDs are global, not namespaced by study.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np


def union_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def subtract_intervals(intervals, exclusions):
    result = []
    for start, end in union_intervals(intervals):
        cursor = start
        for left, right in union_intervals(exclusions):
            if right <= cursor:
                continue
            if left >= end:
                break
            if left > cursor:
                result.append((cursor, min(left, end)))
            cursor = max(cursor, right)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return result


def speaker_blocks(segments, duration, speaker_override=None, max_gap_seconds=60.0):
    """Fill same-speaker gaps, then remove ALL non-primary speech/overlap.

    Select by unioned, in-bounds speaking duration BEFORE adding gaps. All other
    labels are exclusion regions, including a third speaker. Unlabeled gaps at
    speaker changes and leading/trailing silence are not assigned to the subject.
    Gaps strictly longer than max_gap_seconds are excluded, creating separate
    blocks on either side; gaps exactly at the threshold remain intact.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and positive")
    if not math.isfinite(max_gap_seconds) or max_gap_seconds < 0:
        raise ValueError("Maximum gap must be finite and nonnegative")
    by_speaker = defaultdict(list)
    clipped_count = 0
    for start, end, speaker in segments:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid diarization interval")
        if not str(speaker).strip():
            raise ValueError("Empty speaker label")
        clipped_count += int(end > duration)
        if start < duration:
            by_speaker[str(speaker)].append((start, min(end, duration)))
    by_speaker = {k: union_intervals(v) for k, v in by_speaker.items()}
    if not by_speaker:
        raise ValueError("No in-bounds diarized speech")
    totals = {k: sum(b - a for a, b in v) for k, v in by_speaker.items()}
    ranked = sorted(totals, key=lambda k: (-totals[k], k))
    primary = ranked[0] if speaker_override is None else speaker_override
    if primary not in by_speaker:
        raise ValueError("Speaker override is absent from in-bounds diarization")
    own = by_speaker[primary]
    other = union_intervals([v for k, vs in by_speaker.items() if k != primary for v in vs])
    expanded = list(own)
    gaps = []
    excluded_gaps = []
    for (_, left_end), (right_start, _) in zip(own, own[1:]):
        if not any(a < right_start and b > left_end for a, b in other):
            if right_start - left_end > max_gap_seconds:
                excluded_gaps.append((left_end, right_start))
            else:
                expanded.append((left_end, right_start))
                gaps.append(right_start - left_end)
    blocks = subtract_intervals(expanded, other)
    speech_seconds = sum(b - a for a, b in subtract_intervals(own, other))
    return primary, blocks, {
        "speaker_durations": totals,
        "speaker_count": len(totals),
        "primary_speech_share": totals[primary] / sum(totals.values()),
        "primary_duration_tie": len(ranked) > 1 and math.isclose(totals[ranked[0]], totals[ranked[1]]),
        "diarization_segments_clipped": clipped_count,
        "max_preserved_gap_seconds": max(gaps, default=0.0),
        "excluded_long_gaps": excluded_gaps,
        "excluded_long_gap_seconds": sum(b - a for a, b in excluded_gaps),
        "removed_overlap_seconds": totals[primary] - speech_seconds,
        "retained_speech_seconds": speech_seconds,
        "retained_block_seconds": sum(b - a for a, b in blocks),
    }


def assign_words(words, blocks):
    """Keep only fully contained words; omit ambiguous speaker-boundary words.

    Zero-duration words are retained at start <= t < end. Returned timestamps
    stay absolute. No word is assigned to more than one block.
    """
    assigned = [[] for _ in blocks]
    for word in words:
        for index, (start, end) in enumerate(blocks):
            if start <= word["start"] < end and word["end"] <= end:
                assigned[index].append(word)
                break
    return assigned


def block_windows(start, end, words, max_seconds=55.0):
    """Partition a block without overlap; keep short blocks AND short tails.

    Move an artificial boundary backward to a word start if needed, to avoid
    cutting a word. Reject a timestamped word longer than the window limit.
    Each word occurs once; duration weights cannot double-count overlapping time.
    """
    if not math.isfinite(max_seconds) or max_seconds <= 0 or end <= start:
        raise ValueError("Invalid window size or block interval")
    result = []
    cursor = start
    while cursor < end - 1e-9:
        stop = min(end, cursor + max_seconds)
        while True:
            crossing = [w for w in words if w["start"] < stop < w["end"]]
            if not crossing:
                break
            stop = min(w["start"] for w in crossing)
            if stop <= cursor + 1e-9:
                raise ValueError("Word or overlapping word group exceeds window duration")
        current = [w for w in words if cursor <= w["start"] < stop and w["end"] <= stop]
        result.append((cursor, stop, current))
        cursor = stop
    return result


def stable_id(value):
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def loso_folds(recordings):
    """Outer study holdouts plus inner study holdouts; purge shared subjects."""
    studies = sorted({r["study"] for r in recordings})
    if len(studies) < 3:
        raise ValueError("Nested study holdouts require at least three studies")
    result = []
    for study in studies:
        test = [r for r in recordings if r["study"] == study]
        test_subjects = {r["subject_id"] for r in test}
        train = [r for r in recordings if r["subject_id"] not in test_subjects]
        inner = []
        for validation_study in sorted({r["study"] for r in train}):
            validation = [r for r in train if r["study"] == validation_study]
            validation_subjects = {r["subject_id"] for r in validation}
            fit = [r for r in train if r["subject_id"] not in validation_subjects]
            if not fit or not validation:
                raise ValueError("Empty inner partition after subject purging")
            inner.append({"validation_study": validation_study, "train": fit, "validation": validation})
        result.append({"test_study": study, "train": train, "test": test, "inner": inner})
    return result


def mixed_subject_folds(recordings, n_splits=5, seed=40):
    """Deterministic study-balanced subject folds, independent of row order/labels.

    Multi-study subjects are grouped by their complete study-membership tuple.
    Balancing counts subjects, not visits. This is a secondary in-mixture test.
    """
    membership = defaultdict(set)
    for row in recordings:
        membership[row["subject_id"]].add(row["study"])
    strata = defaultdict(list)
    for subject, studies in membership.items():
        strata[tuple(sorted(studies))].append(subject)
    assignment = {}
    offset = 0
    for stratum, subjects in sorted(strata.items()):
        subjects.sort(key=lambda subject: stable_id(f"{seed}:{subject}"))
        for index, subject in enumerate(subjects):
            assignment[subject] = (index + offset) % n_splits
        offset = (offset + len(subjects)) % n_splits
    return [
        {"fold": i, "train": [r for r in recordings if assignment[r["subject_id"]] != i],
         "test": [r for r in recordings if assignment[r["subject_id"]] == i]}
        for i in range(n_splits)
    ]


def regression_metrics(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.ndim != 1 or truth.shape != prediction.shape or not len(truth):
        raise ValueError("Expected nonempty matching vectors")
    if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("Nonfinite targets or predictions")
    error = prediction - truth
    sst = float(np.square(truth - truth.mean()).sum())
    prediction_sst = float(np.square(prediction - prediction.mean()).sum())
    variable = len(truth) > 1 and sst > 1e-12 and prediction_sst > 1e-12
    slope = float(np.dot(prediction - prediction.mean(), truth - truth.mean()) / prediction_sst) if prediction_sst > 1e-12 else None
    return {
        "recordings": len(truth), "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": 1 - float(np.square(error).sum()) / sst if sst > 1e-12 else None,
        "pearson_r": float(np.corrcoef(truth, prediction)[0, 1]) if variable else None,
        "mean_prediction_error": float(error.mean()),
        "calibration_slope": slope,
        "calibration_intercept": float(truth.mean() - slope * prediction.mean()) if slope is not None else None,
        "madrs_mean": float(truth.mean()), "madrs_sd": float(truth.std()),
        "madrs_min": float(truth.min()), "madrs_max": float(truth.max()),
    }


def load_window_audio(window, sample_rate=16000):
    """Read ONE contiguous crop, mono/resample it, return waveform and true length.

    Do not repeat short audio. A model-specific collator may zero-pad it and must
    supply the appropriate attention mask. Never use padding for pause features.
    """
    import soundfile as sf
    import torch
    import torchaudio

    with sf.SoundFile(window["audio_path"]) as audio:
        rate = audio.samplerate
        first = math.ceil(window["start_seconds"] * rate)
        last = min(len(audio), math.floor(window["end_seconds"] * rate))
        if last <= first:
            raise ValueError("Window is shorter than one source audio sample")
        audio.seek(first)
        waveform = torch.from_numpy(audio.read(last - first, dtype="float32", always_2d=True).mean(axis=1))
    if rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, rate, sample_rate)
    return waveform, waveform.numel()
