from __future__ import annotations

import torch
from torch import nn


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def add_awgn(
    waveforms: torch.Tensor, snr_min_db: float, snr_max_db: float
) -> torch.Tensor:
    """Add zero-mean white noise at an independently sampled SNR per waveform."""
    if snr_min_db > snr_max_db:
        raise ValueError("snr_min_db cannot exceed snr_max_db")
    snr_db = torch.empty(
        waveforms.shape[0], 1, device=waveforms.device, dtype=waveforms.dtype
    ).uniform_(snr_min_db, snr_max_db)
    noise = torch.randn_like(waveforms)
    signal_power = waveforms.square().mean(dim=1, keepdim=True).clamp_min(1e-12)
    noise_power = noise.square().mean(dim=1, keepdim=True).clamp_min(1e-12)
    target_noise_power = signal_power / torch.pow(10.0, snr_db / 10.0)
    return waveforms + noise * torch.sqrt(target_noise_power / noise_power)


def add_synthetic_reverb(
    waveforms: torch.Tensor,
    sample_rate: int,
    rt60_min_seconds: float,
    rt60_max_seconds: float,
    wet_min: float,
    wet_max: float,
) -> torch.Tensor:
    """Convolve waveforms with randomized exponentially decaying synthetic RIRs."""
    if rt60_min_seconds <= 0 or rt60_min_seconds > rt60_max_seconds:
        raise ValueError("Expected 0 < rt60_min_seconds <= rt60_max_seconds")
    if not 0.0 <= wet_min <= wet_max <= 1.0:
        raise ValueError("Expected 0 <= wet_min <= wet_max <= 1")

    batch_size, waveform_length = waveforms.shape
    rir_length = max(2, round(rt60_max_seconds * sample_rate))
    times = torch.arange(
        rir_length, device=waveforms.device, dtype=waveforms.dtype
    ).unsqueeze(0) / sample_rate
    rt60 = torch.empty(
        batch_size, 1, device=waveforms.device, dtype=waveforms.dtype
    ).uniform_(rt60_min_seconds, rt60_max_seconds)
    decay = torch.pow(10.0, -3.0 * times / rt60)
    tail = torch.randn_like(decay) * decay
    tail[:, 0] = 0.0
    tail = tail / tail.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
    tail_gain = torch.empty(
        batch_size, 1, device=waveforms.device, dtype=waveforms.dtype
    ).uniform_(0.25, 0.75)
    impulse_response = tail * tail_gain
    impulse_response[:, 0] = 1.0

    convolution_length = waveform_length + rir_length - 1
    fft_length = 1 << (convolution_length - 1).bit_length()
    waveform_fft = torch.fft.rfft(waveforms, n=fft_length)
    response_fft = torch.fft.rfft(impulse_response, n=fft_length)
    wet_waveforms = torch.fft.irfft(
        waveform_fft * response_fft, n=fft_length
    )[:, :waveform_length]

    dry_rms = waveforms.square().mean(dim=1, keepdim=True).sqrt()
    wet_rms = wet_waveforms.square().mean(dim=1, keepdim=True).sqrt()
    wet_waveforms = wet_waveforms * (dry_rms / wet_rms.clamp_min(1e-12))
    wet = torch.empty(
        batch_size, 1, device=waveforms.device, dtype=waveforms.dtype
    ).uniform_(wet_min, wet_max)
    return waveforms * (1.0 - wet) + wet_waveforms * wet


class WaveformAugmenter(nn.Module):
    """Apply independent AWGN and synthetic-reverb transforms to a batch."""

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        awgn_probability: float = 0.5,
        awgn_snr_min_db: float = 10.0,
        awgn_snr_max_db: float = 30.0,
        reverb_probability: float = 0.3,
        reverb_rt60_min_seconds: float = 0.2,
        reverb_rt60_max_seconds: float = 0.8,
        reverb_wet_min: float = 0.1,
        reverb_wet_max: float = 0.4,
    ) -> None:
        super().__init__()
        _validate_probability("awgn_probability", awgn_probability)
        _validate_probability("reverb_probability", reverb_probability)
        if awgn_snr_min_db > awgn_snr_max_db:
            raise ValueError("awgn_snr_min_db cannot exceed awgn_snr_max_db")
        if (
            reverb_rt60_min_seconds <= 0
            or reverb_rt60_min_seconds > reverb_rt60_max_seconds
        ):
            raise ValueError(
                "Expected 0 < reverb_rt60_min_seconds <= reverb_rt60_max_seconds"
            )
        if not 0.0 <= reverb_wet_min <= reverb_wet_max <= 1.0:
            raise ValueError("Expected 0 <= reverb_wet_min <= reverb_wet_max <= 1")
        self.sample_rate = sample_rate
        self.awgn_probability = awgn_probability
        self.awgn_snr_min_db = awgn_snr_min_db
        self.awgn_snr_max_db = awgn_snr_max_db
        self.reverb_probability = reverb_probability
        self.reverb_rt60_min_seconds = reverb_rt60_min_seconds
        self.reverb_rt60_max_seconds = reverb_rt60_max_seconds
        self.reverb_wet_min = reverb_wet_min
        self.reverb_wet_max = reverb_wet_max

    @staticmethod
    def _selection(
        batch_size: int, probability: float, device: torch.device
    ) -> torch.Tensor:
        return torch.rand(batch_size, device=device) < probability

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError("Expected waveforms with shape [batch, samples]")
        augmented = waveforms

        reverb_selection = self._selection(
            waveforms.shape[0], self.reverb_probability, waveforms.device
        )
        if reverb_selection.any():
            augmented = augmented.clone()
            augmented[reverb_selection] = add_synthetic_reverb(
                augmented[reverb_selection],
                self.sample_rate,
                self.reverb_rt60_min_seconds,
                self.reverb_rt60_max_seconds,
                self.reverb_wet_min,
                self.reverb_wet_max,
            )

        awgn_selection = self._selection(
            waveforms.shape[0], self.awgn_probability, waveforms.device
        )
        if awgn_selection.any():
            if augmented is waveforms:
                augmented = augmented.clone()
            augmented[awgn_selection] = add_awgn(
                augmented[awgn_selection],
                self.awgn_snr_min_db,
                self.awgn_snr_max_db,
            )
        return augmented
