"""Biomedical Signal Processing loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "cross_entropy",
    "bce_with_logits",
    "class_balanced_focal",
    "supervised_contrastive",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
