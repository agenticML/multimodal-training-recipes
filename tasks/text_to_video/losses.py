"""Text-to-Video loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "noise_mse",
    "v_prediction_mse",
    "min_snr_mse",
    "temporal_consistency",
    "text_video_alignment",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
