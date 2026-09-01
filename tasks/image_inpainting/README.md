# Image Inpainting

Five pinned architecture recipes for **masked image generation**.

## Architectures

- `stable_diffusion_inpaint` — Stable Diffusion Inpaint; [pinned Hub source](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/tree/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb)
- `sdxl_inpaint` — SDXL Inpaint; [pinned Hub source](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/tree/115134f363124c53c7d878647567d04daf26e41e)
- `kandinsky22_inpaint` — Kandinsky 2.2 Inpaint; [pinned Hub source](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint/tree/db790ad5cbcabed886f069ef2710774657621702)
- `repaint` — RePaint; [pinned Hub source](https://huggingface.co/google/ddpm-ema-celebahq-256/tree/4cb6117472e6e4f45c5afe606b101858c27c3802)
- `controlnet_inpaint` — ControlNet Inpaint; [pinned Hub source](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint/tree/c96e03a807e64135568ba8aecb66b3a306ec73bd)

## Losses

- `masked_noise_mse`
- `masked_charbonnier`
- `perceptual_lpips`
- `ssim_boundary`
- `context_l1`

## Run

```bash
./tasks/image_inpainting/train.sh --architecture stable_diffusion_inpaint --profile smoke
./tasks/image_inpainting/train.sh --architecture stable_diffusion_inpaint --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `masked_PSNR`, `masked_SSIM`, `LPIPS`, `FID`.
