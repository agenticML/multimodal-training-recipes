"""Unconditional Image Generation loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "epsilon_mse",
    "v_prediction_mse",
    "huber_diffusion",
    "min_snr_mse",
    "adversarial_feature_matching",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
