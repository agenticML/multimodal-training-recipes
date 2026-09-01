"""Audio Super Resolution loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "waveform_l1",
    "multi_resolution_stft",
    "si_sdr",
    "log_mel_spectral_convergence",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
