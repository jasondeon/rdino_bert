import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mentalbert_evaluation import (
    MentalBertRegressor, TextWindow, adaptation_state, aggregate_chunks,
    restore_adaptation, sample_recording_windows, select_epoch,
    target_standardization, tokenize_windows,
)
from scripts.build_evaluation_bundle import build, write_csv
from scripts.run_mentalbert_loso import main, validate_subjects


def tiny_tokenizer(directory):
    vocabulary = directory / "vocab.txt"
    vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ni\nfeel\nbetter\ntoday\n")
    return BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=True)


class MentalBertDataTests(unittest.TestCase):
    def test_token_overflow_is_preserved_and_weighted(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = tiny_tokenizer(Path(directory))
            rows = [{"window_id": "w", "recording_id": "r", "duration_seconds": "9",
                     "word_count": "9", "text": "i feel better today i feel better today i"}]
            window = tokenize_windows(rows, tokenizer, max_tokens=6)["w"]
            self.assertEqual([len(c["input_ids"]) for c in window.chunks], [6, 6, 3])
            self.assertEqual(sum(len(c["input_ids"]) - 2 for c in window.chunks), 9)
            np.testing.assert_allclose(window.chunk_weights, [4/9, 4/9, 1/9])

    def test_target_scaling_uses_only_supplied_training_records(self):
        self.assertEqual(target_standardization([{"regression_label": "10"}, {"regression_label": "20"}]),
                         {"mean": 15.0, "std": 5.0})

    def test_sampling_is_reproducible_and_proportional_to_duration(self):
        records = [{"recording_id": "r", "regression_label": "10"}]
        short = TextWindow("s", "r", 1, [], [])
        long = TextWindow("l", "r", 9, [], [])
        grouped = {"r": [short, long]}
        first = sample_recording_windows(records, grouped, 10000, 40, 1)[0][1]
        second = sample_recording_windows(records, grouped, 10000, 40, 1)[0][1]
        self.assertEqual([w.window_id for w in first], [w.window_id for w in second])
        self.assertAlmostEqual(sum(w.window_id == "l" for w in first)/10000, .9, delta=.02)

    def test_recording_aggregation_is_differentiable(self):
        predictions = torch.tensor([1., 3., 10.], requires_grad=True)
        pooled = aggregate_chunks(predictions, torch.tensor([0, 0, 1]), torch.tensor([.25, .75, 1]), 2)
        torch.testing.assert_close(pooled, torch.tensor([2.5, 10.]))
        pooled.sum().backward()
        torch.testing.assert_close(predictions.grad, torch.tensor([.25, .75, 1.]))

    def test_epoch_selection_uses_equal_study_rmse_and_earlier_ties(self):
        epoch, _ = select_epoch([
            [{"epoch": 5, "rmse": 2}, {"epoch": 10, "rmse": 1}, {"epoch": 15, "rmse": 1}],
            [{"epoch": 5, "rmse": 6}, {"epoch": 10, "rmse": 5}, {"epoch": 15, "rmse": 5}],
        ])
        self.assertEqual(epoch, 10)

    def test_subject_leakage_rejected(self):
        with self.assertRaises(ValueError):
            validate_subjects([{"subject_id": "same"}], [], [{"subject_id": "same"}])

    def test_adaptation_checkpoint_includes_normalization_buffers(self):
        backbone = BertModel(BertConfig(vocab_size=9, hidden_size=8, num_hidden_layers=1,
                                       num_attention_heads=2, intermediate_size=16))
        model = MentalBertRegressor("unused", None, rank=2, alpha=2, backbone=backbone)
        model.normalization.running_mean.fill_(4)
        state = adaptation_state(model)
        self.assertIn("normalization.running_mean", state)
        self.assertTrue(any("lora_A" in name for name in state))
        self.assertFalse(any("word_embeddings.weight" in name for name in state))
        model.normalization.running_mean.zero_()
        restore_adaptation(model, state)
        torch.testing.assert_close(model.normalization.running_mean, torch.full((8,), 4.))


class TinyCpuExperimentTests(unittest.TestCase):
    def test_full_nested_run_resume_and_scoring(self):
        """Exercise the real runner with six synthetic recordings and a tiny BERT."""
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                model_dir = root / "tiny_model"
                model_dir.mkdir()
                tokenizer = tiny_tokenizer(model_dir)
                tokenizer.save_pretrained(model_dir)
                BertModel(BertConfig(vocab_size=9, hidden_size=8, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=16,
                                     max_position_embeddings=32)).save_pretrained(model_dir)
                rows = []
                for index in range(6):
                    audio = root / f"audio{index}.wav"
                    words = root / f"words{index}.json"
                    diarization = root / f"diarization{index}.json"
                    sf.write(audio, np.zeros(400), 100)
                    words.write_text(json.dumps([{"word": "i", "start": 1, "end": 1.2},
                                                 {"word": "feel", "start": 1.3, "end": 1.5}]))
                    diarization.write_text(json.dumps([{"start": 0, "end": 4, "speaker": "P"}]))
                    rows.append({"audio_path": str(audio), "word_timestamps_path": str(words),
                                 "diarization_path": str(diarization), "regression_label": 5 + index * 3,
                                 "subject_id": f"p{index}", "study": "ABC"[index//2]})
                source = root / "source.csv"
                write_csv(source, rows)
                bundle, output = root / "bundle", root / "experiment"
                build(argparse.Namespace(source=source, output=bundle, speaker_overrides=None,
                                         window_seconds=55., seed=40, max_gap_seconds=60.))
                argv = ["run", "--bundle", str(bundle), "--output", str(output), "--model", str(model_dir),
                        "--device", "cpu", "--max-epochs", "1", "--validation-interval", "1",
                        "--max-tokens", "16", "--lora-rank", "2", "--lora-alpha", "2",
                        "--windows-per-recording", "2", "--recordings-per-batch", "1"]
                with patch("sys.argv", argv):
                    main()
                metrics = json.loads((output / "metrics.json").read_text())
                self.assertEqual(metrics["recordings"], 6)
                self.assertEqual(len(metrics["by_study"]), 3)
                checkpoint = output / "A" / "final" / "last_checkpoint.pt"
                before = checkpoint.stat().st_mtime_ns
                with patch("sys.argv", argv):
                    main()
                self.assertEqual(checkpoint.stat().st_mtime_ns, before)
                # Simulate interruption after a durable final checkpoint but before
                # completion marker. Resume must finish without rerunning epoch 1.
                (output / "A" / "complete.json").unlink()
                (output / "A" / "final" / "complete.json").unlink()
                with patch("sys.argv", argv + ["--study", "A"]):
                    main()
                self.assertEqual(checkpoint.stat().st_mtime_ns, before)
                with patch("sys.argv", argv + ["--learning-rate", "0.02"]):
                    with self.assertRaises(ValueError):
                        main()
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
