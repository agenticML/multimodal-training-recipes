# Large Language Models

Five pinned architecture recipes for **causal language modeling**.

## Architectures

- `gpt2` — GPT-2; [pinned Hub source](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)
- `llama` — Llama; [pinned Hub source](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/fe8a4ea1ffedaf415f4da2f062534de366a451e6)
- `mistral` — Mistral; [pinned Hub source](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/27d67f1b5f57dc0953326b2601d68371d40ea8da)
- `gemma` — Gemma 2; [pinned Hub source](https://huggingface.co/google/gemma-2-2b/tree/c5ebcd40d208330abc697524c919956e692655cf)
- `qwen` — Qwen2.5; [pinned Hub source](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987)

## Losses

- `causal_cross_entropy`
- `label_smoothed_cross_entropy`
- `focal_token`
- `unlikelihood`

## Run

```bash
./tasks/llm/train.sh --architecture gpt2 --profile smoke
./tasks/llm/train.sh --architecture gpt2 --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `perplexity`, `token_accuracy`, `exact_match`.
