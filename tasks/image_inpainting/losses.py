"""Image Inpainting loss adapters."""

from training_recipes.losses import compute_loss

SUPPORTED_LOSSES = [
    "masked_noise_mse",
    "masked_charbonnier",
    "perceptual_lpips",
    "ssim_boundary",
    "context_l1",
]
__all__ = ["SUPPORTED_LOSSES", "compute_loss"]
