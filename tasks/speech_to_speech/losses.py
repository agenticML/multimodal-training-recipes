"""Speech-to-Speech loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "mel_mse",
    "mel_l1",
    "multi_resolution_stft",
    "si_sdr",
    "content_speaker_contrastive",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
