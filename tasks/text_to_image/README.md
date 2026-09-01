# Text-to-Image

Five pinned architecture recipes for **conditional image generation**.

## Architectures

- `stable_diffusion` — Stable Diffusion; [pinned Hub source](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14)
- `sdxl` — Stable Diffusion XL; [pinned Hub source](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/462165984030d82259a11f4367a4eed129e94a7b)
- `kandinsky22` — Kandinsky 2.2; [pinned Hub source](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder/tree/9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3)
- `pixart_alpha` — PixArt-Alpha; [pinned Hub source](https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512/tree/50f702106901db6d0f8b67eb88e814c56ded2692)
- `stable_cascade` — Stable Cascade; [pinned Hub source](https://huggingface.co/stabilityai/stable-cascade/tree/a89f66d459ae653e3b4d4f992a7c3789d0dc4d16)

## Losses

- `epsilon_mse`
- `v_prediction_mse`
- `min_snr_mse`
- `prior_preservation`
- `text_image_contrastive`

## Run

```bash
./tasks/text_to_image/train.sh --architecture stable_diffusion --profile smoke
./tasks/text_to_image/train.sh --architecture stable_diffusion --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `FID`, `KID`, `CLIP_score`, `prompt_alignment`.
