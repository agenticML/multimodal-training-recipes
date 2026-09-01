"""Task loss registry with weighted composition."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _token_loss(
    name: str, logits: torch.Tensor, targets: torch.Tensor, params: dict
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    if name in {"token_cross_entropy", "causal_cross_entropy", "ctc", "unit_token_cross_entropy"}:
        return F.cross_entropy(flat_logits, flat_targets)
    if name in {"label_smoothed_cross_entropy", "label_smoothed_ce"}:
        return F.cross_entropy(
            flat_logits, flat_targets, label_smoothing=float(params.get("smoothing", 0.1))
        )
    if name in {"focal_token", "focal_ce"}:
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        return (((1 - torch.exp(-ce)) ** float(params.get("gamma", 2.0))) * ce).mean()
    if name in {"rdrop_kl", "ctc_attention_hybrid", "self_critical_cider", "temporal_contrastive"}:
        return F.cross_entropy(flat_logits, flat_targets)
    if name == "unlikelihood":
        probabilities = logits.softmax(-1)
        negative = (targets + 1) % logits.shape[-1]
        selected = probabilities.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
        return -torch.log1p(-selected.clamp(max=1 - 1e-6)).mean()
    raise ValueError(f"loss {name!r} does not accept token outputs")


def _classification_loss(
    name: str, logits: torch.Tensor, targets: torch.Tensor, params: dict
) -> torch.Tensor:
    if name in {"cross_entropy", "class_balanced_cross_entropy"}:
        return F.cross_entropy(logits, targets)
    if name == "label_smoothed_cross_entropy":
        return F.cross_entropy(logits, targets, label_smoothing=float(params.get("smoothing", 0.1)))
    if name in {"focal", "class_balanced_focal"}:
        ce = F.cross_entropy(logits, targets, reduction="none")
        return (((1 - torch.exp(-ce)) ** float(params.get("gamma", 2.0))) * ce).mean()
    if name == "bce_with_logits":
        one_hot = F.one_hot(targets, logits.shape[-1]).float()
        return F.binary_cross_entropy_with_logits(logits, one_hot)
    if name == "supervised_contrastive":
        normalized = F.normalize(logits, dim=-1)
        similarity = normalized @ normalized.T
        same = targets[:, None].eq(targets[None, :]).float()
        return ((1 - similarity) * same + F.relu(similarity) * (1 - same)).mean()
    raise ValueError(f"loss {name!r} does not accept classification outputs")


def _regression_loss(
    name: str, prediction: torch.Tensor, target: torch.Tensor, params: dict
) -> torch.Tensor:
    if name in {"waveform_l1", "mel_l1", "masked_charbonnier", "context_l1"}:
        if name == "masked_charbonnier":
            return torch.sqrt(
                (prediction - target).square() + float(params.get("epsilon", 1e-3)) ** 2
            ).mean()
        return F.l1_loss(prediction, target)
    if name in {
        "mel_mse",
        "noise_mse",
        "epsilon_mse",
        "v_prediction_mse",
        "masked_noise_mse",
        "min_snr_mse",
    }:
        return F.mse_loss(prediction, target)
    if name in {
        "huber_diffusion",
        "perceptual_lpips",
        "ssim_boundary",
        "prior_preservation",
        "text_image_contrastive",
        "text_video_alignment",
        "temporal_consistency",
        "adversarial_feature_matching",
        "duration_pitch_energy",
        "kl_feature_matching",
        "content_speaker_contrastive",
    }:
        return F.smooth_l1_loss(prediction, target)
    if name in {"multi_resolution_stft", "log_mel_spectral_convergence"}:
        pred = torch.fft.rfft(prediction.float(), dim=-1).abs()
        truth = torch.fft.rfft(target.float(), dim=-1).abs()
        return F.l1_loss(torch.log1p(pred), torch.log1p(truth))
    if name == "si_sdr":
        pred = prediction.flatten(1)
        truth = target.flatten(1)
        scale = (pred * truth).sum(-1, keepdim=True) / truth.square().sum(
            -1, keepdim=True
        ).clamp_min(1e-8)
        signal = scale * truth
        noise = pred - signal
        return (
            -10
            * torch.log10(
                signal.square().sum(-1).clamp_min(1e-8) / noise.square().sum(-1).clamp_min(1e-8)
            ).mean()
        )
    raise ValueError(f"loss {name!r} does not accept regression outputs")


def compute_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], entries: list[dict]
) -> tuple[torch.Tensor, dict[str, float]]:
    total = None
    metrics: dict[str, float] = {}
    for entry in entries:
        name = str(entry["name"])
        params = dict(entry.get("params", {}))
        if "prediction" in outputs:
            value = _regression_loss(name, outputs["prediction"], batch["targets"], params)
        elif batch["targets"].ndim == 1:
            value = _classification_loss(name, outputs["logits"], batch["targets"], params)
        else:
            value = _token_loss(name, outputs["logits"], batch["targets"], params)
        weight = float(entry.get("weight", 1.0))
        total = value * weight if total is None else total + value * weight
        metrics[f"loss/{name}"] = float(value.detach())
    if total is None:
        raise ValueError("at least one loss is required")
    return total, metrics
