"""Text-to-Image loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "epsilon_mse",
    "v_prediction_mse",
    "min_snr_mse",
    "prior_preservation",
    "text_image_contrastive",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
