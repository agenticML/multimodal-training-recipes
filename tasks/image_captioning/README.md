# Image Captioning

Five pinned architecture recipes for **image-to-text**.

## Architectures

- `vit_gpt2` — ViT-GPT2 VisionEncoderDecoder; [pinned Hub source](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning/tree/dc68f91c06a1ba6f15268e5b9c13ae7a7c514084)
- `git` — GIT; [pinned Hub source](https://huggingface.co/microsoft/git-base/tree/1f7fe8444292beb4a259e3a5b6eba440cd5999d4)
- `blip` — BLIP; [pinned Hub source](https://huggingface.co/Salesforce/blip-image-captioning-base/tree/82a37760796d32b1411fe092ab5d4e227313294b)
- `blip2` — BLIP-2; [pinned Hub source](https://huggingface.co/Salesforce/blip2-opt-2.7b/tree/59a1ef6c1e5117b3f65523d1c6066825bcf315e3)
- `instructblip` — InstructBLIP; [pinned Hub source](https://huggingface.co/Salesforce/instructblip-flan-t5-xl/tree/bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d)

## Losses

- `token_cross_entropy`
- `label_smoothed_cross_entropy`
- `focal_token`
- `self_critical_cider`

## Run

```bash
./tasks/image_captioning/train.sh --architecture vit_gpt2 --profile smoke
./tasks/image_captioning/train.sh --architecture vit_gpt2 --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `CIDEr`, `BLEU_4`, `METEOR`, `ROUGE_L`.
