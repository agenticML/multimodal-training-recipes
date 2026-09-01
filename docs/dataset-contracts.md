# Dataset contracts

All real-data profiles consume a UTF-8 JSONL manifest. Relative paths are resolved from the manifest directory. The loader accepts one of these normalized record forms:

```json
{"input_ids": [1, 8, 4], "targets": [8, 4, 2]}
{"inputs": [[0.1, 0.2]], "targets": [[0.1, 0.25]]}
{"tensor_path": "samples/000001.pt"}
```

A `.pt` record must be a dictionary containing `inputs` and `targets` tensors. Preprocessing raw media into this normalized form keeps decoding, augmentation, and model training reproducible.

| Tasks | Canonical tensors |
| --- | --- |
| ASR, translation, captions, LLM | Token IDs and target token IDs; production adapters may replace input IDs with processor-produced audio/pixel tensors. |
| TTS, speech-to-speech, audio super resolution | `[channels, samples]` source and target audio. |
| Image generation/inpainting/text-to-image | `[3, height, width]`; inpainting preprocessors concatenate image, mask, and masked-image channels as required upstream. |
| Video generation/text-to-video | `[3, frames, height, width]`. |
| Image/video classification | Media tensor plus integer or one-hot label. |
| Biomedical signal processing | `[channels, samples]`, modality metadata (`ecg`, `eeg`, or `ppg`), sample rate, channel names, and label(s). |

The synthetic profile constructs the same canonical tensor families with a fixed per-sample seed. It is a plumbing and differentiability check, not a quality benchmark.
