"""Official LeRobot policy, strict weights and one train/deploy processor pair."""
from pathlib import Path

import torch
from torch.utils.data import default_collate

from .config import BASE, EVIDENCE, VLM_CONFIG
from .runtime import read


def load(directory=BASE, device="cuda", frozen_bf16=True):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    cfg = SmolVLAConfig.from_pretrained(directory, local_files_only=True)
    cfg.device = "cpu"
    cfg.vlm_model_name = str(VLM_CONFIG)
    cfg.load_vlm_weights = False
    policy = SmolVLAPolicy(cfg).to(dtype=torch.float32)
    # Avoid permissive defaults in from_pretrained: every key must match.
    from safetensors.torch import load_model
    missing, unexpected = load_model(policy, str(Path(directory) / "model.safetensors"), strict=True, device="cpu")
    assert not missing and not unexpected
    policy.to(device)
    variant_path = Path(directory) / "training-variant.json"
    variant = read(variant_path) if variant_path.exists() else None
    if frozen_bf16 and device == "cuda":
        # Base VLM values originally came from BF16. Only frozen parameters use
        # BF16 storage; trainable expert/state projection retain FP32 masters.
        # Preserve fine-tuned FP32 tail values before casting frozen modules.
        preserved = {name: {k: v.detach().clone() for k, v in policy.get_submodule(name).state_dict().items()}
                     for name in variant.get("fp32_modules", [])} if variant else {}
        policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
        for name, values in preserved.items():
            module = policy.get_submodule(name).to(dtype=torch.float32)
            module.load_state_dict(values, strict=True)
    if variant:
        policy._training_variant = variant
    policy.config.device = device
    policy.eval()
    return policy


def processors(policy, directory=None):
    from lerobot.policies.factory import make_pre_post_processors
    raw = read(EVIDENCE / "train-only-stats.json")
    stats = {k: {n: torch.tensor(v, dtype=torch.float32) for n, v in values.items()} for k, values in raw.items()}
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, dataset_stats=stats, pretrained_path=directory,
        preprocessor_overrides={"device_processor": {"device": policy.config.device},
                                "tokenizer_processor": {"tokenizer_name": str(Path(directory) / "tokenizer") if directory and (Path(directory) / "tokenizer").is_dir() else policy.config.vlm_model_name}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    for pipeline, key in [(pre, "observation.state"), (post, "action")]:
        normalizers = [s for s in pipeline.steps if hasattr(s, "stats") and key in s.stats]
        assert len(normalizers) == 1
        for name in ["mean", "std"]:
            torch.testing.assert_close(torch.as_tensor(normalizers[0].stats[key][name]).cpu(), stats[key][name], rtol=0, atol=0)
    pre._smolvla_image_crop = getattr(policy,"_training_variant",{}).get("image_crop_xyxy")
    return pre, post, stats


def batch(samples, pre):
    crop = getattr(pre,"_smolvla_image_crop",None)
    if crop:
        from .config import IMAGE
        left,top,right,bottom=crop
        samples=[dict(s,**{IMAGE:s[IMAGE][:,top:bottom,left:right].contiguous()}) for s in samples]
        assert all(s[IMAGE].shape == (3,bottom-top,right-left) for s in samples)
    return pre(default_collate(samples))


def train_mode(policy):
    policy.train()
    policy.model.vlm_with_expert.set_requires_grad()
    assert not policy.model.vlm_with_expert.vlm.training
    assert all(not p.requires_grad for p in policy.model.vlm_with_expert.vlm.parameters())
    if hasattr(policy, "_training_variant"):
        for name in policy._training_variant["fp32_modules"]:
            policy.get_submodule(name).requires_grad_(True)

