# Unconditional Video Generation

Five pinned architecture recipes for **null-conditioned video generation**.

## Architectures

- `video_diffusion_unet3d` — Video Diffusion UNet3D; [pinned Hub source](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b/tree/8227dddca75a8561bf858d604cc5dae52b954d01)
- `animatediff_unconditional` — AnimateDiff Null-Conditioned; [pinned Hub source](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3/tree/2e8139b1d1269fd8a21deb96ad19455e187692eb)
- `latte_dit` — Latte Video-DiT; [pinned Hub source](https://huggingface.co/maxin-cn/Latte-1/tree/0653024365272f061fc44d1078134df22842b687)
- `cogvideox_unconditional` — CogVideoX Null-Conditioned; [pinned Hub source](https://huggingface.co/THUDM/CogVideoX-2b/tree/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01)
- `ltx_video_unconditional` — LTX-Video Null-Conditioned; [pinned Hub source](https://huggingface.co/Lightricks/LTX-Video/tree/8984fa25007f376c1a299016d0957a37a2f797bb)

## Losses

- `noise_mse`
- `v_prediction_mse`
- `min_snr_mse`
- `temporal_consistency`
- `adversarial_feature_matching`

## Run

```bash
./tasks/video_generation/train.sh --architecture video_diffusion_unet3d --profile smoke
./tasks/video_generation/train.sh --architecture video_diffusion_unet3d --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `FVD`, `temporal_LPIPS`, `inception_score`, `motion_smoothness`.
