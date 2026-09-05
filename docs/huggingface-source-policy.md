# Hugging Face source policy

Every architecture record contains a Hub repository, immutable 40-character revision, library, public class, upstream architecture source locator, and explicit `trust_remote_code` flag. The project never silently executes remote code and does not vendor third-party modeling files.

`scripts/resolve_hf_sources.py` compares every pin with the current Hub head. Revision drift is reported but never applied automatically. Review the upstream diff, model card, license, and security implications before changing a pin.

Smoke recipes instantiate the pinned Hub class from a tiny config (`load_mode: tiny_config`) and dummy task-shaped tensors. Dummy runs never call `from_pretrained` and must work offline from library defaults plus size overrides. Custom and mlx recipes (`audiosr`, `flashsr`, `nuwave2`, `aero`, `universr`, `stylegan2_ada`) have no importable Hub class and are skipped. Full configurations identify the actual upstream class, checkpoint, loss, and fine-tuning policy. Large generative and multimodal models default to adapter/LoRA fine-tuning in their configuration.
