"""Video Classification loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = ["cross_entropy", "label_smoothed_cross_entropy", "focal", "bce_with_logits"]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
