# Automatic Speech Recognition

Five pinned architecture recipes for **audio-to-token**.

## Architectures

- `wav2vec2` — Wav2Vec2; [pinned Hub source](https://huggingface.co/facebook/wav2vec2-base-960h/tree/22aad52d435eb6dbaf354bdad9b0da84ce7d6156)
- `hubert` — HuBERT; [pinned Hub source](https://huggingface.co/facebook/hubert-large-ls960-ft/tree/ece5fabbf034c1073acae96d5401b25be96709d8)
- `data2vec_audio` — Data2Vec Audio; [pinned Hub source](https://huggingface.co/facebook/data2vec-audio-base-960h/tree/32331f3123e703528918aa688a9a38232d58c872)
- `whisper` — Whisper; [pinned Hub source](https://huggingface.co/openai/whisper-tiny/tree/169d4a4341b33bc18d8881c4b69c2e104e1cc0af)
- `speech2text` — Speech2Text; [pinned Hub source](https://huggingface.co/facebook/s2t-small-librispeech-asr/tree/256e91a603c013ae09aa1e4c09cfc3f8acdf8d19)

## Losses

- `ctc`
- `token_cross_entropy`
- `label_smoothed_cross_entropy`
- `focal_token`
- `ctc_attention_hybrid`

## Run

```bash
./tasks/asr/train.sh --architecture wav2vec2 --profile smoke
./tasks/asr/train.sh --architecture wav2vec2 --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `WER`, `CER`, `token_accuracy`.
