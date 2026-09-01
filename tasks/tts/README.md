# Text-to-Speech

Five pinned architecture recipes for **audio regression**.

## Architectures

- `speecht5` — SpeechT5; [pinned Hub source](https://huggingface.co/microsoft/speecht5_tts/tree/30fcde30f19b87502b8435427b5f5068e401d5f6)
- `vits_mms` — VITS / MMS-TTS; [pinned Hub source](https://huggingface.co/facebook/mms-tts-eng/tree/c71de0fe7204c83f1c10820a7d696d0b450048ba)
- `fastspeech2_conformer` — FastSpeech2-Conformer; [pinned Hub source](https://huggingface.co/espnet/kan-bayashi_csmsc_conformer_fastspeech2/tree/130e1e771be8670de6b420691cc755373673fcbc)
- `bark` — Bark; [pinned Hub source](https://huggingface.co/suno/bark-small/tree/1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd)
- `seamless_m4t_t2s` — SeamlessM4Tv2 T2S; [pinned Hub source](https://huggingface.co/facebook/seamless-m4t-v2-large/tree/5f8cc790b19fc3f67a61c105133b20b34e3dcb76)

## Losses

- `mel_l1`
- `mel_mse`
- `multi_resolution_stft`
- `duration_pitch_energy`
- `kl_feature_matching`

## Run

```bash
./tasks/tts/train.sh --architecture speecht5 --profile smoke
./tasks/tts/train.sh --architecture speecht5 --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `mel_cepstral_distortion`, `PESQ`, `STOI`, `speaker_similarity`.
