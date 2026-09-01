# Video Captioning

Five pinned architecture recipes for **video-to-text**.

## Architectures

- `git_vatex` — GIT VATEX; [pinned Hub source](https://huggingface.co/microsoft/git-base-vatex/tree/623363cef8744c57438ce8d62b30bef1fbbec111)
- `video_llava` — Video-LLaVA; [pinned Hub source](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf/tree/4cf9d8cfc76a54f46a4cb43be5368b46b7f0d736)
- `llava_next_video` — LLaVA-NeXT-Video; [pinned Hub source](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf/tree/f417d9edeb69a9eaa88283780332d00ce3c66593)
- `instructblip_video` — InstructBLIP Video; [pinned Hub source](https://huggingface.co/Salesforce/instructblip-flan-t5-xl/tree/bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d)
- `videollama3` — VideoLLaMA3; [pinned Hub source](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B/tree/d5b763e368861e7f5096e7ff1b49f92fbccf8ae6)

## Losses

- `token_cross_entropy`
- `label_smoothed_cross_entropy`
- `focal_token`
- `self_critical_cider`
- `temporal_contrastive`

## Run

```bash
./tasks/video_captioning/train.sh --architecture git_vatex --profile smoke
./tasks/video_captioning/train.sh --architecture git_vatex --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `CIDEr`, `BLEU_4`, `METEOR`, `temporal_grounding`.
