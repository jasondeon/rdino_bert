from __future__ import annotations

import argparse
from pathlib import Path

from dataset import read_diarization, read_manifest, read_word_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multimodal CSV manifests")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--skip-audio-path-check", action="store_true")
    args = parser.parse_args()

    train = read_manifest(args.train, args.num_classes)
    validation = read_manifest(args.validation, args.num_classes)
    if not args.skip_audio_path_check:
        missing = [
            path
            for item in train + validation
            for path in (item.audio_path, item.word_timestamps_path, item.diarization_path)
            if not path.is_file()
        ]
        if missing:
            preview = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(f"{len(missing)} input files are missing. First paths:\n{preview}")
        for item in train + validation:
            read_word_timestamps(item.word_timestamps_path)
            read_diarization(item.diarization_path)

    train_subjects = {item.subject_id for item in train if item.subject_id}
    validation_subjects = {item.subject_id for item in validation if item.subject_id}
    overlap = train_subjects & validation_subjects
    if overlap:
        preview = ", ".join(sorted(overlap)[:10])
        raise ValueError(f"Subject leakage across splits ({len(overlap)} subjects): {preview}")

    print(f"Train recordings: {len(train)}")
    print(f"Validation recordings: {len(validation)}")
    print("Manifest validation passed")


if __name__ == "__main__":
    main()
