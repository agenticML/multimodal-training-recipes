# Audio Super Resolution

Five pinned architecture recipes for **low-to-high bandwidth audio**.

## Architectures

- `audiosr` — AudioSR; [pinned Hub source](https://huggingface.co/haoheliu/audiosr_basic/tree/74a47f49061a1e788e968cc43ad45c0b6243f37d)
- `flashsr` — FlashSR; [pinned Hub source](https://huggingface.co/aufklarer/FlashSR-MLX-8bit/tree/63da0bc604ed355b0ce9d91d8d2fbf5995743768)
- `nuwave2` — NU-Wave2; [pinned Hub source](https://huggingface.co/huseinzol05/nuwave2/tree/d025c7f418884a36f5634ec69e9219a3b9ce8816)
- `aero` — AERO; [pinned Hub source](https://huggingface.co/oulianov/slp-rl-aero/tree/33ab062aedb14cb5723313f6699f5694f0c11b52)
- `universr` — UniverSR; [pinned Hub source](https://huggingface.co/woongzip1/universr-audio/tree/1c3294844285af851b6ffa56cbde4e43cd41fc2b)

## Losses

- `waveform_l1`
- `multi_resolution_stft`
- `si_sdr`
- `log_mel_spectral_convergence`

## Run

```bash
./tasks/audio_super_resolution/train.sh --architecture audiosr --profile smoke
./tasks/audio_super_resolution/train.sh --architecture audiosr --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `SI_SDR`, `log_spectral_distance`, `PESQ`, `STOI`.
