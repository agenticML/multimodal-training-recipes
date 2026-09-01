# Model matrix

The authoritative machine-readable inventory is each task's `architectures.yaml`. Every entry pins the Hub repository to an immutable commit and records the upstream class and modeling-source locator.

| Task | Architecture recipe names | Default profile loss |
| --- | --- | --- |
| TTS | `speecht5`, `vits_mms`, `fastspeech2_conformer`, `bark`, `seamless_m4t_t2s` | `mel_l1` |
| ASR | `wav2vec2`, `hubert`, `data2vec_audio`, `whisper`, `speech2text` | `ctc` |
| Speech-to-Speech | `speecht5_vc`, `seamless_m4t`, `seamless_m4t_v2`, `moshi`, `qwen_omni` | `mel_l1` |
| Translation | `marian`, `m2m100`, `mbart50`, `nllb`, `t5` | `token_cross_entropy` |
| Image Generation | `ddpm_unet2d`, `dit`, `consistency_model`, `vqgan_transformer`, `stylegan2_ada` | `epsilon_mse` |
| Image Inpainting | `stable_diffusion_inpaint`, `sdxl_inpaint`, `kandinsky22_inpaint`, `repaint`, `controlnet_inpaint` | `masked_noise_mse` |
| Video Generation | `video_diffusion_unet3d`, `animatediff_unconditional`, `latte_dit`, `cogvideox_unconditional`, `ltx_video_unconditional` | `noise_mse` |
| Text-to-Image | `stable_diffusion`, `sdxl`, `kandinsky22`, `pixart_alpha`, `stable_cascade` | `epsilon_mse` |
| Text-to-Video | `animatediff`, `cogvideox`, `modelscope_t2v`, `ltx_video`, `mochi` | `noise_mse` |
| Image Captioning | `vit_gpt2`, `git`, `blip`, `blip2`, `instructblip` | `token_cross_entropy` |
| Video Captioning | `git_vatex`, `video_llava`, `llava_next_video`, `instructblip_video`, `videollama3` | `token_cross_entropy` |
| LLM | `gpt2`, `llama`, `mistral`, `gemma`, `qwen` | `causal_cross_entropy` |
| Biomedical | `patchtst`, `patchtsmixer`, `time_series_transformer`, `informer`, `autoformer` | `cross_entropy` |
| Video Classification | `videomae`, `timesformer`, `vivit`, `mvit`, `xclip` | `cross_entropy` |
| Image Classification | `vit`, `deit`, `swin`, `convnext`, `beit` | `cross_entropy` |
| Audio Super Resolution | `audiosr`, `flashsr`, `nuwave2`, `aero`, `universr` | `waveform_l1` |

Loss compatibility is intentionally task-level rather than universal. For example, ASR CTC heads and encoder-decoder heads use different native objectives, while diffusion recipes distinguish epsilon and velocity targets. The task manifests are the source of truth for accepted `--loss` overrides.
