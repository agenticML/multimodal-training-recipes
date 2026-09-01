# Speech-to-Speech

Five pinned architecture recipes for **audio-to-audio**.

## Architectures

- `speecht5_vc` — SpeechT5 Voice Conversion; [pinned Hub source](https://huggingface.co/microsoft/speecht5_vc/tree/c418ba2144598f973d0fddc9fd5909a3af83de3d)
- `seamless_m4t` — SeamlessM4T; [pinned Hub source](https://huggingface.co/facebook/hf-seamless-m4t-medium/tree/ecf60d4df63baaac3f82ae6a7ad7adcb19dcb26c)
- `seamless_m4t_v2` — SeamlessM4Tv2; [pinned Hub source](https://huggingface.co/facebook/seamless-m4t-v2-large/tree/5f8cc790b19fc3f67a61c105133b20b34e3dcb76)
- `moshi` — Moshi; [pinned Hub source](https://huggingface.co/kyutai/moshika-pytorch-bf16/tree/a49141e28b3d9c947cf9aa5314431e1b11cbd2f5)
- `qwen_omni` — Qwen2.5-Omni; [pinned Hub source](https://huggingface.co/Qwen/Qwen2.5-Omni-3B/tree/f75b40e3da2003cdd6e1829b1f420ca70797c34e)

## Losses

- `mel_mse`
- `mel_l1`
- `multi_resolution_stft`
- `si_sdr`
- `content_speaker_contrastive`

## Run

```bash
./tasks/speech_to_speech/train.sh --architecture speecht5_vc --profile smoke
./tasks/speech_to_speech/train.sh --architecture speecht5_vc --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `ASR_BLEU`, `speaker_similarity`, `PESQ`, `STOI`.
