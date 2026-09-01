# Machine Translation

Five pinned architecture recipes for **text-to-text**.

## Architectures

- `marian` — MarianMT; [pinned Hub source](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr/tree/dd7f6540a7a48a7f4db59e5c0b9c42c8eea67f18)
- `m2m100` — M2M100; [pinned Hub source](https://huggingface.co/facebook/m2m100_418M/tree/55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636)
- `mbart50` — mBART-50; [pinned Hub source](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt/tree/e30b6cb8eb0d43a0b73cab73c7676b9863223a30)
- `nllb` — NLLB; [pinned Hub source](https://huggingface.co/facebook/nllb-200-distilled-600M/tree/f8d333a098d19b4fd9a8b18f94170487ad3f821d)
- `t5` — T5; [pinned Hub source](https://huggingface.co/google-t5/t5-small/tree/df1b051c49625cf57a3d0d8d3863ed4d13564fe4)

## Losses

- `token_cross_entropy`
- `label_smoothed_cross_entropy`
- `focal_token`
- `rdrop_kl`

## Run

```bash
./tasks/translation/train.sh --architecture marian --profile smoke
./tasks/translation/train.sh --architecture marian --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `BLEU`, `chrF`, `COMET`.
