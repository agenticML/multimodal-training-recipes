# Biomedical Signal Processing

Five pinned architecture recipes for **ECG EEG PPG classification**.

## Architectures

- `patchtst` — PatchTST; [pinned Hub source](https://huggingface.co/namctin/patchtst_etth1_pretrain/tree/5dcaab0b40fa2b8a3843e2197e1f6b6e53ecd363)
- `patchtsmixer` — PatchTSMixer; [pinned Hub source](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/tree/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f)
- `time_series_transformer` — Time Series Transformer; [pinned Hub source](https://huggingface.co/huggingface/time-series-transformer-tourism-monthly/tree/2a40ad41f6ffe61e7bef6099b08c6c2fce36ac35)
- `informer` — Informer; [pinned Hub source](https://huggingface.co/huggingface/informer-tourism-monthly/tree/da30269a25658b3626dda20a94c9bf6b765f1b13)
- `autoformer` — Autoformer; [pinned Hub source](https://huggingface.co/huggingface/autoformer-tourism-monthly/tree/3801324c8213b225c5ee93ee53f7bfd8094ae7e2)

## Losses

- `cross_entropy`
- `bce_with_logits`
- `class_balanced_focal`
- `supervised_contrastive`

## Run

```bash
./tasks/biomedical_signal_processing/train.sh --architecture patchtst --profile smoke
./tasks/biomedical_signal_processing/train.sh --architecture patchtst --profile full --dataset-path /path/to/train.jsonl
```

The real-data input is a UTF-8 JSONL manifest described in [dataset contracts](../../docs/dataset-contracts.md). Evaluation reports `macro_F1`, `AUROC`, `AUPRC`, `sensitivity`, `specificity`.
