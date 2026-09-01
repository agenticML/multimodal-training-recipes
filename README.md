# Multimodal Training Recipes

This project provides 80 reproducible training recipes across the 16 requested speech, language, image, video, biomedical, and restoration tasks. Every task contains five Hugging Face architecture references, four or more configurable losses, optimizer choices, a dataset adapter, per-architecture configurations, a shell launcher, and a runbook.

## Install

Python 3.10 or newer and PyTorch are required. From this directory:

```bash
uv sync --extra dev
```

Install Hugging Face and media integrations when running full recipes:

```bash
uv sync --extra all --extra dev
```

The lightweight smoke suite does not download pretrained weights. It uses small task-contract models to validate data shapes, configured objectives, backward propagation, optimization, and artifacts for all 80 recipes. Full configs carry the pinned upstream model class and fine-tuning policy and require the modality dependencies, real preprocessed data, and suitable accelerators.

## Run one recipe

Every task launcher accepts the same options:

```bash
./tasks/asr/train.sh --architecture wav2vec2 --profile smoke

./tasks/asr/train.sh \
  --architecture wav2vec2 \
  --profile full \
  --dataset-path /data/asr/train.jsonl \
  --device cuda \
  --max-steps 1000
```

Use a chosen Python environment without activating it:

```bash
PYTHON=/path/to/venv/bin/python ./tasks/translation/train.sh \
  --architecture nllb --profile smoke
```

List architectures or override training fields:

```bash
./tasks/text_to_image/train.sh --list
./tasks/image_classification/train.sh --architecture vit --profile smoke \
  --loss focal --optimizer lion --learning-rate 0.0003 --max-steps 5
```

Outputs are written to `outputs/<task>/<architecture>/` and include the resolved configuration, JSONL metrics, summary, and checkpoint.

Resume or force cached/offline model resolution with:

```bash
./tasks/llm/train.sh --architecture gpt2 --profile smoke \
  --resume-from outputs/llm/gpt2/checkpoint.pt --max-steps 10
./tasks/llm/train.sh --architecture gpt2 --profile smoke --offline
```

## Validate and test everything

```bash
uv run python scripts/validate_catalog.py
uv run pytest
uv run ./scripts/smoke_all.sh --steps 1
```

The complete smoke command executes one real forward/backward/optimizer step for all 80 configurations. Restrict it to one task with `--task video_classification` or retain artifacts with `--output-dir /path/to/output`.

To detect whether an upstream repository has moved past the reviewed commit:

```bash
uv run python scripts/resolve_hf_sources.py
```

The command reports revision drift and never rewrites locks automatically.

## Task index

| Task directory | Scope |
| --- | --- |
| `tts` | Text-to-speech synthesis |
| `asr` | Automatic speech recognition |
| `speech_to_speech` | Voice conversion and speech translation |
| `translation` | Text-to-text machine translation |
| `image_generation` | Unconditional image generation |
| `image_inpainting` | Mask-conditioned image restoration/generation |
| `video_generation` | Unconditional or null-conditioned video generation |
| `text_to_image` | Prompt-conditioned image generation |
| `text_to_video` | Prompt-conditioned video generation |
| `image_captioning` | Image-to-text captioning |
| `video_captioning` | Video-to-text captioning |
| `llm` | Causal language modeling and SFT-style token training |
| `biomedical_signal_processing` | ECG, EEG, and PPG classification |
| `video_classification` | Single-label or multilabel video classification |
| `image_classification` | Single-label or multilabel image classification |
| `audio_super_resolution` | Audio bandwidth extension and restoration |

See [dataset contracts](docs/dataset-contracts.md), [evaluation](docs/evaluation.md), [the model matrix](docs/model-matrix.md), and [the Hugging Face source policy](docs/huggingface-source-policy.md).

## Important operating notes

- A full profile requires `--dataset-path` and defaults to CUDA/bfloat16. Override device, batch size, and step count for the available hardware.
- Large diffusion, video, and multimodal models require substantial VRAM; their configs default to LoRA where appropriate.
- Model licenses remain those of their upstream repositories. Review them before commercial use or redistribution.
- `trust_remote_code` is recorded explicitly for custom architectures; inspect pinned source before enabling it in an upstream loader.
