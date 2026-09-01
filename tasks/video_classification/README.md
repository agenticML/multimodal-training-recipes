# Video Classification

Five pinned architecture recipes for **video classification**.

## Architectures

- `videomae` — VideoMAE; [pinned Hub source](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics/tree/488eb9a0565f257b32866000305c8178965eb9f6)
- `timesformer` — TimeSformer; [pinned Hub source](https://huggingface.co/facebook/timesformer-base-finetuned-k400/tree/8aaf40ea7d3d282dcb0a5dea01a198320d15d6c0)
- `vivit` — ViViT; [pinned Hub source](https://huggingface.co/google/vivit-b-16x2-kinetics400/tree/8a7171a57f79b9aaa58bc8d977c002a0ea0f0d42)
- `mvit` — MViTv2; [pinned Hub source](https://huggingface.co/DanJoshua/mvit_v2_s_Kinetics400_transf_b_rwf2000/tree/b0f764ceabfe894db9c5a96ce605a3e363476d18)
- `xclip` — X-CLIP; [pinned Hub source](https://huggingface.co/microsoft/xclip-base-patch32/tree/a2e27a78a2b5d802e894b8a1ef14f3a8ce490963)

## Losses

- `cross_entropy`
- `label_smoothed_cross_entropy`
- `focal`
- `bce_with_logits`

## Run

```bash
./tasks/video_classification/train.sh --architecture videomae --profile smoke
./tasks/video_classification/train.sh --architecture videomae --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `top1_accuracy`, `top5_accuracy`, `macro_F1`, `mAP`.
