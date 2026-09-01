"""Text-to-Speech loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "mel_l1",
    "mel_mse",
    "multi_resolution_stft",
    "duration_pitch_energy",
    "kl_feature_matching",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
