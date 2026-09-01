"""Video Captioning loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "token_cross_entropy",
    "label_smoothed_cross_entropy",
    "focal_token",
    "self_critical_cider",
    "temporal_contrastive",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
