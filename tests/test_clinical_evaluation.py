import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from clinical_evaluation import (
    assign_words, block_windows, load_window_audio, loso_folds,
    mixed_subject_folds, regression_metrics, speaker_blocks,
)
from scripts.score_evaluation_bundle import collect_predictions


class SpeakerBoundaryTests(unittest.TestCase):
    def test_pause_preserved_and_interviewer_breaks_block(self):
        speaker, blocks, _ = speaker_blocks(
            [(1, 3, "P"), (5, 9, "P"), (10, 11, "I"), (12, 20, "P")], 25,
        )
        self.assertEqual(speaker, "P")
        self.assertEqual(blocks, [(1, 9), (12, 20)])

    def test_gap_at_sixty_seconds_is_preserved_but_longer_gap_splits(self):
        _, blocks, qc = speaker_blocks([(0, 10, "P"), (70, 80, "P"), (141, 151, "P")], 151)
        self.assertEqual(blocks, [(0, 80), (141, 151)])
        self.assertEqual(qc["max_preserved_gap_seconds"], 60)
        self.assertEqual(qc["excluded_long_gaps"], [(80, 141)])
        self.assertEqual(qc["excluded_long_gap_seconds"], 61)

    def test_long_gap_does_not_join_across_interviewer(self):
        _, blocks, qc = speaker_blocks([(0, 20, "P"), (30, 31, "I"), (90, 120, "P")], 120)
        self.assertEqual(blocks, [(0, 20), (90, 120)])
        self.assertEqual(qc["excluded_long_gaps"], [])

    def test_nested_and_crossing_other_speaker_are_removed(self):
        _, blocks, qc = speaker_blocks(
            [(0, 20, "P"), (5, 7, "I"), (18, 23, "I"), (22, 30, "P")], 30,
        )
        self.assertEqual(blocks, [(0, 5), (7, 18), (23, 30)])
        self.assertEqual(qc["removed_overlap_seconds"], 5)

    def test_choose_duration_not_turn_count_or_gap_length(self):
        speaker, _, _ = speaker_blocks(
            [(0, 1, "I"), (2, 3, "I"), (100, 101, "I"), (110, 120, "P")], 120,
        )
        self.assertEqual(speaker, "P")

    def test_same_speaker_overlap_does_not_double_count_duration(self):
        speaker, _, qc = speaker_blocks([(0, 8, "A"), (0, 8, "A"), (10, 20, "B")], 20)
        self.assertEqual(speaker, "B")
        self.assertEqual(qc["speaker_durations"]["A"], 8)

    def test_clip_before_speaker_selection(self):
        speaker, blocks, qc = speaker_blocks([(0, 8, "P"), (9, 1000, "I")], 10)
        self.assertEqual(speaker, "P")
        self.assertEqual(blocks, [(0, 8)])
        self.assertEqual(qc["diarization_segments_clipped"], 1)

    def test_boundary_words_removed_and_timestamps_preserved(self):
        words = [{"word": "ok", "start": 1, "end": 2},
                 {"word": "ambiguous", "start": 2.5, "end": 3.5},
                 {"word": "interviewer", "start": 3, "end": 4},
                 {"word": "after", "start": 5, "end": 6}]
        assigned = assign_words(words, [(0, 3), (5, 8)])
        self.assertEqual(assigned, [[words[0]], [words[3]]])

    def test_short_blocks_tails_and_word_boundary_no_duplication(self):
        words = [{"word": "boundary", "start": 54, "end": 56},
                 {"word": "end", "start": 58, "end": 59}]
        windows = block_windows(0, 60, words, 55)
        self.assertEqual([(a, b) for a, b, _ in windows], [(0, 54), (54, 60)])
        self.assertEqual(sum(len(ws) for _, _, ws in windows), 2)
        self.assertEqual(block_windows(5, 6, [], 55), [(5, 6, [])])

    def test_audio_crop_keeps_original_silence_and_never_joins_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            waveform = np.r_[np.ones(10), np.zeros(10), np.ones(10), -np.ones(10)]
            sf.write(path, waveform, 10, subtype="FLOAT")
            audio, length = load_window_audio({"audio_path": str(path), "start_seconds": 0, "end_seconds": 3}, sample_rate=10)
            self.assertEqual(length, 30)
            np.testing.assert_array_equal(audio.numpy(), waveform[:30])


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"recording_id": f"{s}{i}{j}", "subject_id": f"{s}{i}", "study": s}
                     for s in "ABCDE" for i in range(7) for j in range(2)]

    def test_outer_and_inner_subject_disjoint_and_each_test_once(self):
        tested = []
        for fold in loso_folds(self.rows):
            train = {r["subject_id"] for r in fold["train"]}
            test = {r["subject_id"] for r in fold["test"]}
            self.assertFalse(train & test)
            tested.extend(r["recording_id"] for r in fold["test"])
            for inner in fold["inner"]:
                fit = {r["subject_id"] for r in inner["train"]}
                val = {r["subject_id"] for r in inner["validation"]}
                self.assertFalse(fit & val or fit & test or val & test)
        self.assertCountEqual(tested, [r["recording_id"] for r in self.rows])

    def test_cross_study_subject_is_purged(self):
        self.rows.append({"recording_id": "shared", "subject_id": "A0", "study": "B"})
        fold = loso_folds(self.rows)[0]
        self.assertNotIn("shared", {r["recording_id"] for r in fold["train"]})

    def test_mixed_folds_stable_under_row_permutation(self):
        a = mixed_subject_folds(self.rows)
        b = mixed_subject_folds(list(reversed(self.rows)))
        for left, right in zip(a, b):
            self.assertEqual({r["recording_id"] for r in left["test"]}, {r["recording_id"] for r in right["test"]})
            self.assertFalse({r["subject_id"] for r in left["train"]} & {r["subject_id"] for r in left["test"]})


class ScoringTests(unittest.TestCase):
    def test_constant_target_and_prediction_are_undefined(self):
        metrics = regression_metrics([3, 3], [2, 2])
        self.assertIsNone(metrics["r2"])
        self.assertIsNone(metrics["pearson_r"])
        self.assertEqual(metrics["mean_prediction_error"], -1)

    def test_calibration_distinguishes_offset_from_association(self):
        metrics = regression_metrics([1, 2, 3], [11, 12, 13])
        self.assertAlmostEqual(metrics["pearson_r"], 1)
        self.assertAlmostEqual(metrics["calibration_intercept"], -10)
        self.assertLess(metrics["r2"], 0)

    def test_aggregation_and_strict_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "recordings.csv").write_text("recording_id,study\nr,A\n")
            (bundle / "windows.csv").write_text("window_id,recording_id,aggregation_weight\nw1,r,0.25\nw2,r,0.75\n")
            rows = [{"window_id": "w1", "test_study": "A", "prediction": "0"},
                    {"window_id": "w2", "test_study": "A", "prediction": "20"}]
            _, predictions = collect_predictions(bundle, rows, "window")
            self.assertEqual(predictions["r"], 15)
            for invalid in (rows[:1], rows + rows[:1], [rows[0], dict(rows[1], test_study="B")]):
                with self.assertRaises(ValueError):
                    collect_predictions(bundle, invalid, "window")


    def test_text_view_renormalizes_without_changing_recording_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "recordings.csv").write_text("recording_id,study\nr,A\n")
            (bundle / "windows.csv").write_text("window_id,recording_id,aggregation_weight,word_count\nw1,r,0.25,0\nw2,r,0.75,2\n")
            rows = [{"window_id": "w2", "test_study": "A", "prediction": "20"}]
            _, predictions = collect_predictions(bundle, rows, "window", "text")
            self.assertEqual(predictions["r"], 20)
            with self.assertRaises(ValueError):
                collect_predictions(bundle, rows, "window", "all")


if __name__ == "__main__":
    unittest.main()
