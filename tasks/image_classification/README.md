# Image Classification

Five pinned architecture recipes for **image classification**.

## Architectures

- `vit` — Vision Transformer; [pinned Hub source](https://huggingface.co/google/vit-base-patch16-224/tree/3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3)
- `deit` — DeiT; [pinned Hub source](https://huggingface.co/facebook/deit-base-patch16-224/tree/fb2c78a54a5637dec350432794f7b93e31f910c9)
- `swin` — Swin Transformer; [pinned Hub source](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/tree/d00d478bfaf1f417d34a1186673ad4b4be264c51)
- `convnext` — ConvNeXT; [pinned Hub source](https://huggingface.co/facebook/convnext-tiny-224/tree/6166b7613034066690a621d8bf25ffdf181a34f0)
- `beit` — BEiT; [pinned Hub source](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k/tree/9da301148150e37e533abef672062fa49f6bda4f)

## Losses

- `cross_entropy`
- `label_smoothed_cross_entropy`
- `focal`
- `bce_with_logits`

## Run

```bash
./tasks/image_classification/train.sh --architecture vit --profile smoke
./tasks/image_classification/train.sh --architecture vit --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `top1_accuracy`, `top5_accuracy`, `macro_F1`, `AUROC`.
