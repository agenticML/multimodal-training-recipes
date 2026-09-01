"""Unconditional Video Generation loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "noise_mse",
    "v_prediction_mse",
    "min_snr_mse",
    "temporal_consistency",
    "adversarial_feature_matching",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
