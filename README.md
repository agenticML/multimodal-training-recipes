# Multimodal Training Recipes

This project provides 80 reproducible training recipes across the 16 requested speech, language, image, video, biomedical, and restoration tasks. Every task contains five Hugging Face architecture references, four or more configurable losses, optimizer choices, a dataset adapter, per-architecture configurations, a shell launcher, and a runbook.

## Install

Python 3.10 or newer and PyTorch are required. From this directory:

```bash
uv sync --extra dev
```

`transformers` and `diffusers` are main dependencies so dummy/smoke runs can construct Hub classes offline. Install extra media and Hub tooling when running full recipes:

```bash
uv sync --extra all --extra dev
```

Dummy/smoke runs instantiate the pinned Hub class from a tiny random config (`from_config`, no weight download) and train one step on dummy tensors shaped for that architecture. `--architecture wav2vec2` therefore tracks `Wav2Vec2ForCTC`, not a surrogate. Six custom/mlx recipes (`audiosr`, `flashsr`, `nuwave2`, `aero`, `universr`, `stylegan2_ada`) are skipped. Full configs still identify the upstream checkpoint and fine-tuning policy and require real preprocessed data plus suitable accelerators.

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

`--offline` only disables Hugging Face hub downloads. It does not change experiment tracking.

## Experiment tracking

Every CLI run is tracked with [AgenticML](https://agenticml.xyz) (`pip install agenticml-py`, `import agenticml`). The default mode is offline, with journals under `outputs/<task>/<architecture>/agenticml/` next to the usual run artifacts.

```bash
./tasks/asr/train.sh --architecture wav2vec2 --profile smoke
./tasks/asr/train.sh --architecture wav2vec2 --profile smoke --no-track
./tasks/asr/train.sh --architecture wav2vec2 --profile smoke \
  --tracker-mode online --tracker-project multimodal-training-recipes
```

Online mode needs `AGENTICML_API_KEY` (and optionally `AGENTICML_HOST` / `AGENTICML_MODE`). Use `--no-track` or `--tracker-mode disabled` to skip tracking, `--tracker-name` / `--tracker-group` / `--tracker-offline-dir` / `--agenticml-asset-dir` to override identity and the offline journal root.

There is a dedicated offline entrypoint that always uses `--tracker-mode offline` and requires `--agenticml-asset-dir`:

```bash
training-recipe-offline --task asr --architecture wav2vec2 --profile smoke \
  --tracker-mode offline --agenticml-asset-dir /path/to/agenticml-assets
```

## Local smoke dashboard

A localhost FastAPI UI lists catalog tasks and architectures, runs one smoke dummy at a time via `training-recipe-offline`, and shows captured AgenticML code assets.

```bash
# uses the activated venv (VIRTUAL_ENV), else .venv, else python3
./scripts/start_dashboard.sh           # 127.0.0.1:8000
./scripts/start_dashboard.sh 8080
AGENTICML_ASSET_DIR=/tmp/aml-assets ./scripts/start_dashboard.sh 8080
PYTHON=/path/to/venv/bin/python ./scripts/start_dashboard.sh 8080
```

Install FastAPI and Uvicorn in that environment (`uv sync --extra dashboard` or `pip install '.[dashboard]'`).

Profile is locked to `smoke` and tracking to `offline`. Journals land under `AGENTICML_ASSET_DIR` or `agenticml-assets/` at the repo root. Custom/mlx architectures are listed but not runnable.

## Validate and test everything

```bash
uv run python scripts/validate_catalog.py
uv run pytest
uv run ./scripts/smoke_all.sh --steps 1
```

The complete smoke command executes one real forward/backward/optimizer step for the 74 Hub-backed recipes and skips the six custom/mlx architectures. Restrict it to one task with `--task video_classification` or retain artifacts with `--output-dir /path/to/output`. Smoke and `training-recipe-offline` smoke runs do not need `--dataset-path`.

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
