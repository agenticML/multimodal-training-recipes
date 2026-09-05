"""Optimizer factory used by all recipes."""

from __future__ import annotations

import math

import torch


class Lion(torch.optim.Optimizer):
    """Small dependency-free Lion implementation."""

    def __init__(self, params, lr: float, betas=(0.9, 0.99), weight_decay: float = 0.0):
        super().__init__(params, {"lr": lr, "betas": betas, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[parameter]
                momentum = state.setdefault("momentum", torch.zeros_like(parameter))
                beta1, beta2 = group["betas"]
                update = momentum.mul(beta1).add(grad, alpha=1 - beta1)
                parameter.add_(update.sign(), alpha=-group["lr"])
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def create_optimizer(parameters, config: dict) -> torch.optim.Optimizer:
    name = str(config.get("name", "adamw")).lower()
    lr = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("optimizer.learning_rate must be finite and positive")
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if name == "lion":
        return Lion(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adafactor":
        try:
            from transformers.optimization import Adafactor
        except ImportError as exc:
            raise RuntimeError("Adafactor requires transformers") from exc
        return Adafactor(parameters, lr=lr, relative_step=False, scale_parameter=False)
    if name == "adam8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("adam8bit requires bitsandbytes") from exc
        return bnb.optim.AdamW8bit(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer {name!r}")
