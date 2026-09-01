# Evaluation

Training losses and evaluation metrics are separate registries. Losses remain differentiable objectives; evaluation can include non-differentiable decoding and distribution metrics.

- Speech: WER, CER, PESQ, STOI, SI-SDR, mel-cepstral distortion, speaker similarity.
- Translation and captions: BLEU, chrF, COMET, CIDEr, METEOR, and ROUGE-L.
- Images: FID, KID, CLIP score, PSNR, SSIM, and LPIPS.
- Video: FVD, temporal LPIPS, motion consistency, prompt alignment, accuracy, F1, and mAP.
- Language modeling: held-out loss, perplexity, token accuracy, and exact match where appropriate.
- Biomedical signals: macro-F1, AUROC, AUPRC, sensitivity, and specificity, reported per modality and class.
- Audio super resolution: SI-SDR, log-spectral distance, PESQ, and STOI.

Metrics requiring learned evaluators must record the evaluator model ID and immutable revision in the run artifacts.
