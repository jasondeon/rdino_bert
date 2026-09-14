import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from mixed_evaluation import evaluate_mixed, mixed_partitions, select_mixed_epoch
from scripts.build_evaluation_bundle import build, write_csv
from scripts.score_evaluation_bundle import read_csv
from scripts import run_mentalbert_loso as loso
from scripts import run_mentalbert_mixed as mixed


def fixture(root):
    model = root / "tiny_model"
    model.mkdir()
    vocab = model / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ni\nfeel\nbetter\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(model)
    BertModel(BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=16,
                         max_position_embeddings=32)).save_pretrained(model)
    rows = []
    for subject in range(15):
        for visit in range(2):
            name = f"{subject}_{visit}"
            audio, words, diarization = (root / (name+suffix) for suffix in (".wav", ".words.json", ".diar.json"))
            sf.write(audio, np.zeros(400), 100)
            words.write_text(json.dumps([{"word": "i feel better", "start": 1, "end": 2}]))
            diarization.write_text(json.dumps([{"start": 0, "end": 4, "speaker": "P"}]))
            rows.append({"audio_path": str(audio), "word_timestamps_path": str(words),
                         "diarization_path": str(diarization), "subject_id": f"s{subject}",
                         "study": "ABC"[subject//5], "regression_label": 5+2*subject+visit})
    source, bundle, reference = root / "source.csv", root / "bundle", root / "reference"
    write_csv(source, rows)
    build(argparse.Namespace(source=source, output=bundle, speaker_overrides=None,
                             window_seconds=55., seed=40, max_gap_seconds=60.))
    args = SimpleNamespace(bundle=bundle, model=str(model), weights_revision=None, allow_download=False,
                           max_tokens=16, study=None, lora_rank=2, lora_alpha=2, normalization="batchnorm",
                           learning_rate=1e-4, weight_decay=1e-6, max_epochs=1, validation_interval=1,
                           recordings_per_batch=2, windows_per_recording=2, inference_batch_size=8,
                           precision="float32", seed=40)
    config, _, _, _, _ = loso.preflight(args)
    reference.mkdir()
    loso.atomic_json(reference / "experiment_config.json", config)
    records = read_csv(bundle / "recordings.csv")
    write_csv(reference / "loso_recording_predictions.csv", [
        {"recording_id": r["recording_id"], "subject_id": r["subject_id"], "study": r["study"],
         "truth": r["regression_label"], "prediction": float(r["regression_label"])+2} for r in records
    ])
    return bundle, reference


class MixedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.bundle, self.reference = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()
        torch.set_num_threads(self.old_threads)

    def test_scoring_uses_training_only_study_means_and_checks_fold(self):
        records, assignment, folds = mixed_partitions(self.bundle)
        rows = [{"recording_id": r["recording_id"], "test_fold": assignment[r["recording_id"]],
                 "study": r["study"], "prediction": float(r["regression_label"])} for r in records]
        metrics, scored = evaluate_mixed(self.bundle, rows, level="recording")
        self.assertEqual(metrics["pooled"]["mae"], 0)
        self.assertEqual(metrics["subjects"], 15)
        for row in scored:
            train = folds[row["test_fold"]]["train"]
            values = [float(r["regression_label"]) for r in train if r["study"] == row["study"]]
            self.assertAlmostEqual(row["training_study_mean_baseline"], np.mean(values))
        wrong = [dict(r) for r in rows]
        wrong[0]["test_fold"] = "not_the_actual_fold"
        for invalid in (wrong, rows[1:], rows+rows[:1]):
            with self.assertRaises(ValueError):
                evaluate_mixed(self.bundle, invalid, level="recording")

    def test_mixed_partitions_reject_leaked_subject(self):
        fold = self.bundle / "mixed_subject" / "fold_0"
        train, test = read_csv(fold / "train.csv"), read_csv(fold / "test.csv")
        write_csv(fold / "train.csv", train+[test[0]])
        with self.assertRaisesRegex(ValueError, "Subject leakage"):
            mixed_partitions(self.bundle)

    def test_complete_cpu_run_resume_and_comparison(self):
        output = self.root / "experiment"
        argv = ["run", "--bundle", str(self.bundle), "--reference-run", str(self.reference),
                "--output", str(output), "--device", "cpu"]
        with contextlib.redirect_stdout(io.StringIO()), patch("sys.argv", argv):
            mixed.main()
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["recordings"], 30)
        self.assertEqual(len(metrics["by_fold"]), 5)
        self.assertEqual(len(metrics["by_study"]), 3)
        self.assertEqual(len(read_csv(output / "comparison_with_loso.csv")), 4)
        checkpoint = output / "fold_0" / "final" / "last_checkpoint.pt"
        before = checkpoint.stat().st_mtime_ns
        with contextlib.redirect_stdout(io.StringIO()), patch("sys.argv", argv):
            mixed.main()
        self.assertEqual(checkpoint.stat().st_mtime_ns, before)
        (output / "fold_0" / "complete.json").unlink()
        (output / "fold_0" / "inner" / "fold_1" / "complete.json").unlink()
        with contextlib.redirect_stdout(io.StringIO()), patch("sys.argv", argv+["--fold", "fold_0"]):
            mixed.main()
        self.assertEqual(checkpoint.stat().st_mtime_ns, before)
        self.assertTrue((output / "fold_0" / "inner" / "fold_1" / "complete.json").exists())

    def test_inner_optimizer_matches_existing_loso_loop(self):
        args = SimpleNamespace(bundle=self.bundle, reference_run=self.reference, fold=None)
        with contextlib.redirect_stdout(io.StringIO()):
            config, _, tokenizer, grouped, folds = mixed.preflight(args)
            inner = folds["fold_0"]["inner"]["fold_1"]
            for directory, fit in (("old", loso.fit_stage), ("new", mixed.fit_inner_stage)):
                fit(self.root / directory, inner["train"], inner["validation"], grouped,
                    tokenizer, config, torch.device("cpu"), 2, 40)
        old = torch.load(self.root / "old" / "last_checkpoint.pt", weights_only=False)
        new = torch.load(self.root / "new" / "last_checkpoint.pt", weights_only=False)
        for key in old["adaptation"]:
            torch.testing.assert_close(old["adaptation"][key], new["adaptation"][key], rtol=0, atol=0)
        self.assertIn("by_study", new["history"][-1])


class MixedSelectionTests(unittest.TestCase):
    def test_equal_study_selection_not_dominated_by_large_study(self):
        history = [
            {"epoch": 1, "by_study": {"large": {"recordings": 100, "rmse": 1},
                                       "small": {"recordings": 1, "rmse": 9}}},
            {"epoch": 2, "by_study": {"large": {"recordings": 100, "rmse": 4},
                                       "small": {"recordings": 1, "rmse": 4}}},
        ]
        epoch, curve = select_mixed_epoch([history, history])
        self.assertEqual(epoch, 2)
        self.assertEqual(curve[0]["macro_study_rmse"], 5)


if __name__ == "__main__":
    unittest.main()
