# Unconditional Image Generation

Five pinned architecture recipes for **image generation**.

## Architectures

- `ddpm_unet2d` — DDPM / UNet2D; [pinned Hub source](https://huggingface.co/google/ddpm-cifar10-32/tree/267b167dc01f0e4e61923ea244e8b988f84deb80)
- `dit` — Diffusion Transformer; [pinned Hub source](https://huggingface.co/facebook/DiT-XL-2-256/tree/eab87f77abd5aef071a632f08807fbaab0b704d0)
- `consistency_model` — Consistency Model; [pinned Hub source](https://huggingface.co/openai/diffusers-cd_imagenet64_l2/tree/5f462e4403fc37b72ec6004e806c71805db22387)
- `vqgan_transformer` — VQGAN + Transformer; [pinned Hub source](https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384/tree/9f990ec03ea054204706013e2176bbb498ebc387)
- `stylegan2_ada` — StyleGAN2-ADA; [pinned Hub source](https://huggingface.co/matthias-wright/stylegan2/tree/04e43bedd793bf12ba6eeb9048332c6c2ab5f4c6)

## Losses

- `epsilon_mse`
- `v_prediction_mse`
- `huber_diffusion`
- `min_snr_mse`
- `adversarial_feature_matching`

## Run

```bash
./tasks/image_generation/train.sh --architecture ddpm_unet2d --profile smoke
./tasks/image_generation/train.sh --architecture ddpm_unet2d --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `FID`, `KID`, `inception_score`, `LPIPS_diversity`.
