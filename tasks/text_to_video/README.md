# Text-to-Video

Five pinned architecture recipes for **conditional video generation**.

## Architectures

- `animatediff` — AnimateDiff; [pinned Hub source](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3/tree/2e8139b1d1269fd8a21deb96ad19455e187692eb)
- `cogvideox` — CogVideoX; [pinned Hub source](https://huggingface.co/THUDM/CogVideoX-2b/tree/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01)
- `modelscope_t2v` — ModelScope Text-to-Video; [pinned Hub source](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b/tree/8227dddca75a8561bf858d604cc5dae52b954d01)
- `ltx_video` — LTX-Video; [pinned Hub source](https://huggingface.co/Lightricks/LTX-Video/tree/8984fa25007f376c1a299016d0957a37a2f797bb)
- `mochi` — Mochi; [pinned Hub source](https://huggingface.co/genmo/mochi-1-preview/tree/14be5fcea23095ed330cb214647916a451e38b6e)

## Losses

- `noise_mse`
- `v_prediction_mse`
- `min_snr_mse`
- `temporal_consistency`
- `text_video_alignment`

## Run

```bash
./tasks/text_to_video/train.sh --architecture animatediff --profile smoke
./tasks/text_to_video/train.sh --architecture animatediff --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `FVD`, `CLIP_score`, `temporal_LPIPS`, `prompt_alignment`.
