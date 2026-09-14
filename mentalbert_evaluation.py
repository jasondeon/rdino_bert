"""MentalBERT data/model helpers for the frozen clinical evaluation bundle."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModel


@dataclass
class TextWindow:
    window_id: str
    recording_id: str
    duration: float
    chunks: list[dict[str, list[int]]]
    chunk_weights: list[float]


def tokenize_windows(rows, tokenizer, max_tokens=512):
    """Preserve all tokens. Overlength windows become internal token chunks.

    Chunks never cross a canonical window/block boundary. The canonical window
    prediction is the token-count-weighted average of its chunk predictions.
    """
    capacity = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 1:
        raise ValueError("Token limit must leave room for content")
    result = {}
    for row in rows:
        if int(row["word_count"]) == 0:
            continue
        tokens = tokenizer.encode(row["text"], add_special_tokens=False, truncation=False)
        if not tokens:
            raise ValueError(f"Nonempty window has no tokens: {row['window_id']}")
        chunks, weights = [], []
        for start in range(0, len(tokens), capacity):
            part = tokens[start:start + capacity]
            chunks.append(tokenizer.prepare_for_model(
                part, add_special_tokens=True, truncation=False,
                return_attention_mask=True, return_token_type_ids=True,
            ))
            weights.append(len(part) / len(tokens))
        if row["window_id"] in result:
            raise ValueError("Duplicate canonical window ID")
        result[row["window_id"]] = TextWindow(
            row["window_id"], row["recording_id"], float(row["duration_seconds"]), chunks, weights,
        )
    return result


def windows_by_recording(windows):
    result = defaultdict(list)
    for window in windows.values():
        result[window.recording_id].append(window)
    return dict(result)


def target_standardization(records):
    labels = np.asarray([float(r["regression_label"]) for r in records], dtype=float)
    if not len(labels) or not np.isfinite(labels).all():
        raise ValueError("Invalid training labels")
    std = float(labels.std())
    if std < 1e-8:
        raise ValueError("Training MADRS has zero variance")
    return {"mean": float(labels.mean()), "std": std}


def sample_recording_windows(records, grouped_windows, count, seed, epoch):
    """Each recording appears once; draw windows with probability ∝ duration.

    Averaging the draws estimates the duration-weighted recording prediction.
    The finite-draw MSE is a stochastic approximation to the full recording loss.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    result = []
    for index in rng.permutation(len(records)):
        row = records[index]
        windows = grouped_windows[row["recording_id"]]
        probability = np.asarray([w.duration for w in windows], float)
        if not np.isfinite(probability).all() or np.any(probability <= 0):
            raise ValueError("Invalid window duration")
        probability /= probability.sum()
        indices = rng.choice(len(windows), size=count, replace=True, p=probability)
        result.append((row, [windows[i] for i in indices]))
    return result


def collate_recordings(examples, tokenizer, standardization):
    features, owners, weights, targets = [], [], [], []
    for recording_index, (record, windows) in enumerate(examples):
        for window in windows:
            for chunk, weight in zip(window.chunks, window.chunk_weights):
                features.append(chunk)
                owners.append(recording_index)
                weights.append(weight / len(windows))
        targets.append((float(record["regression_label"]) - standardization["mean"]) / standardization["std"])
    tokens = tokenizer.pad(features, padding=True, return_tensors="pt")
    return tokens, torch.tensor(owners), torch.tensor(weights), torch.tensor(targets)


def aggregate_chunks(predictions, owners, weights, count):
    return predictions.new_zeros(count).index_add(0, owners, predictions * weights)


class MentalBertRegressor(nn.Module):
    """Same text pooling and two-layer head as the prior model; no audio/head CE."""
    def __init__(self, model_name, revision, rank=8, alpha=8, normalization="batchnorm",
                 local_files_only=True, backbone=None, weights_revision=None):
        super().__init__()
        if backbone is None:
            config = AutoConfig.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
            # Transformers gives config._commit_hash precedence over revision
            # when resolving cached weights. The converted safetensors live at a
            # separate pinned commit; the architecture still comes from revision.
            config._commit_hash = weights_revision or revision
            backbone = AutoModel.from_pretrained(
                model_name, revision=weights_revision or revision, config=config,
                local_files_only=local_files_only, use_safetensors=True,
            )
        targets = [name for name, module in backbone.named_modules()
                   if isinstance(module, nn.Linear)
                   and (name.endswith(".intermediate.dense") or name.endswith(".output.dense"))
                   and ".attention.output.dense" not in name]
        if not targets:
            raise ValueError("No BERT feed-forward LoRA targets found")
        hidden = backbone.config.hidden_size
        self.backbone = get_peft_model(backbone, LoraConfig(
            r=rank, lora_alpha=alpha, target_modules=targets,
            lora_dropout=.1, bias="none",
        ))
        if normalization not in {"batchnorm", "layernorm"}:
            raise ValueError("Unknown embedding normalization")
        self.normalization = nn.BatchNorm1d(hidden) if normalization == "batchnorm" else nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 200), nn.SiLU(), nn.Dropout(.1),
            nn.Linear(200, 50), nn.SiLU(), nn.Dropout(.1), nn.Linear(50, 1),
        )

    def forward(self, tokens):
        output = self.backbone(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(self.normalization(pooled)).squeeze(-1)


def adaptation_state(model):
    """Save adapters, head, normalization parameters AND running-stat buffers."""
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    names.update(name for name, _ in model.named_buffers())
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name in names}


def restore_adaptation(model, state):
    if set(state) != set(adaptation_state(model)):
        raise ValueError("Checkpoint adaptation keys do not match this model")
    model.load_state_dict(state, strict=False)


def select_epoch(histories):
    """Minimize equal-study RMSE at a common epoch; break ties earlier."""
    if not histories or any(not history for history in histories):
        raise ValueError("Missing inner validation histories")
    maps = [{int(row["epoch"]): float(row["rmse"]) for row in history} for history in histories]
    common = set(maps[0]).intersection(*(set(m) for m in maps[1:]))
    if not common:
        raise ValueError("No common validation epochs")
    curve = [{"epoch": epoch, "macro_study_rmse": float(np.mean([m[epoch] for m in maps]))}
             for epoch in sorted(common)]
    if not all(np.isfinite(row["macro_study_rmse"]) for row in curve):
        raise ValueError("Nonfinite validation score")
    best = min(curve, key=lambda row: (row["macro_study_rmse"], row["epoch"]))
    return best["epoch"], curve
