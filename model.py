from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from speakerlab.utils.builder import build
from speakerlab.utils.config import build_config
from speakerlab.utils.utils import load_params
from torch import nn
from torchaudio import transforms
from transformers import AutoModel


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def _bert_feed_forward_targets(module: nn.Module) -> list[str]:
    """Target both dense layers in each BERT feed-forward block."""
    return [
        name
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
        and (name.endswith(".intermediate.dense") or name.endswith(".output.dense"))
        and ".attention.output.dense" not in name
    ]


def _rdino_feed_forward_targets(module: nn.Module) -> list[str]:
    """Target the terminal attentive-pooling and embedding transformations."""
    candidates = ("asp.tdnn.conv.conv", "asp.conv.conv", "fc.conv")
    available = dict(module.named_modules())
    return [name for name in candidates if isinstance(available.get(name), nn.Conv1d)]


class BertRdinoModel(nn.Module):
    def __init__(
        self,
        *,
        text_model_name: str,
        rdino_yaml: str | Path,
        rdino_checkpoint: str | Path,
        audio_embedding_dim: int = 512,
        fusion_hidden_dim: int = 200,
        fusion_dim: int = 50,
        num_classes: int = 4,
        dropout: float = 0.1,
        lora_rank: int = 2,
        lora_alpha: int = 16,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        text_model = AutoModel.from_pretrained(text_model_name, use_safetensors=True)
        self.text_embedding_dim = text_model.config.hidden_size
        text_targets = _bert_feed_forward_targets(text_model)
        if not text_targets:
            raise RuntimeError("No BERT feed-forward layers found for LoRA")
        self.text_model = get_peft_model(
            text_model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=text_targets,
                lora_dropout=dropout,
                bias="none",
            ),
        )

        rdino_yaml = Path(rdino_yaml).expanduser().resolve()
        rdino_checkpoint = Path(rdino_checkpoint).expanduser().resolve()
        config = build_config(str(rdino_yaml))
        teacher = build("teacher_model", config)
        checkpoint = torch.load(rdino_checkpoint, map_location="cpu", weights_only=False)
        if "teacher" not in checkpoint:
            raise KeyError(f"RDINO checkpoint has no 'teacher' state: {rdino_checkpoint}")
        rdino_model = load_params(teacher, checkpoint["teacher"])
        self.rdino_backbone = rdino_model.backbone

        targets = _rdino_feed_forward_targets(self.rdino_backbone)
        if not targets:
            raise RuntimeError("Expected RDINO pooling/output layers were not found for LoRA")
        self.rdino_backbone = get_peft_model(
            self.rdino_backbone,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=targets,
                lora_dropout=dropout,
                bias="none",
            ),
        )

        self.feature_extractor = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=0.0,
            f_max=8000.0,
            pad=0,
            n_mels=80,
        )
        self.text_normalization = nn.BatchNorm1d(self.text_embedding_dim)
        self.audio_normalization = nn.BatchNorm1d(audio_embedding_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.text_embedding_dim + audio_embedding_dim, fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.regressor = nn.Linear(fusion_dim, 1)
        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for name, parameter in self.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True
        for module in (
            self.text_normalization,
            self.audio_normalization,
            self.fusion,
            self.classifier,
            self.regressor,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(self, tokens: dict[str, torch.Tensor], waveforms: torch.Tensor):
        text_output = self.text_model(**tokens)
        text_embedding = _mean_pool(text_output.last_hidden_state, tokens["attention_mask"])
        text_embedding = self.text_normalization(text_embedding)

        features = self.feature_extractor(waveforms)
        audio_embedding = self.rdino_backbone(features)
        audio_embedding = self.audio_normalization(audio_embedding)

        fused = self.fusion(torch.cat((text_embedding, audio_embedding), dim=1))
        return self.classifier(fused), self.regressor(fused).squeeze(-1)

    def trainable_parameter_counts(self) -> tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return trainable, total
