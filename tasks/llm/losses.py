"""Large Language Models loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "causal_cross_entropy",
    "label_smoothed_cross_entropy",
    "focal_token",
    "unlikelihood",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
