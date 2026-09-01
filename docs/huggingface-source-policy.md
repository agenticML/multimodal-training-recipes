# Hugging Face source policy

Every architecture record contains a Hub repository, immutable 40-character revision, library, public class, upstream architecture source locator, and explicit `trust_remote_code` flag. The project never silently executes remote code and does not vendor third-party modeling files.

`scripts/resolve_hf_sources.py` compares every pin with the current Hub head. Revision drift is reported but never applied automatically. Review the upstream diff, model card, license, and security implications before changing a pin.

Smoke recipes train compact task-contract models so the complete matrix can run without downloading hundreds of gigabytes. Full configurations identify the actual upstream class, checkpoint, loss, and fine-tuning policy. Large generative and multimodal models default to adapter/LoRA fine-tuning in their configuration.
