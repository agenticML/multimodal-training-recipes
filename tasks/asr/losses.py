"""Automatic Speech Recognition loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "ctc",
    "token_cross_entropy",
    "label_smoothed_cross_entropy",
    "focal_token",
    "ctc_attention_hybrid",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
