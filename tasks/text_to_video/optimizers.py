"""Text-to-Video optimizer adapters."""

from training_recipes.optimizers import create_optimizer

SUPPORTED_OPTIMIZERS = ["adamw", "adam", "sgd", "lion", "adafactor", "adam8bit"]
__all__ = ["SUPPORTED_OPTIMIZERS", "create_optimizer"]
