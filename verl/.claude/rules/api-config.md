---
name: Configuration Standards
applies_to: "verl/trainer/config/**, verl/workers/config/**"
---

# verl Configuration Standards

## Hydra YAML Patterns
verl uses Hydra for hierarchical configuration composition.

## Dataclass Conventions
```python
from dataclasses import dataclass, field

@dataclass
class ActorConfig:
    # Required fields first (no default)
    model_path: str

    # Optional fields with defaults
    gradient_checkpointing: bool = False

    # Internal fields last (_prefix)
    _target_: str = "verl.workers.config.ActorConfig"
```

## Field Ordering
1. Required fields (no default)
2. Optional fields (with defaults)
3. Internal fields (`_target_`, `_prefix`)

## Validation
Use `__post_init__` for validation:
```python
def __post_init__(self):
    if self.eps_clip <= 0:
        raise ValueError(f"eps_clip must be positive, got {self.eps_clip}")
```

## YAML Format Rules (CI-enforced)
- Comments above fields (not inline)
- Blank lines between fields
- No trailing whitespace

Example:
```yaml
# Clipping threshold for PPO
eps_clip: 0.2

# KL coefficient
kl_ctl: 0.1
```

## CLI Integration
Fields exposed to CLI need clear help text in docstrings.

## Backward Compatibility
- Adding fields: Safe (add default)
- Removing fields: Requires deprecation warning
- Renaming: Use Union types temporarily

## OmegaConf Features
- Variable interpolation: `${trainer.project_name}`
- Struct mode: Prevents unknown keys
- Plus prefix: `+actor.model.base_model_path=...` adds new keys

## Config Locations

### Hydra YAML Configs (`verl/trainer/config/`)
Organized in subdirectories with YAML files:
- `actor/`: Actor worker configs (`dp_actor.yaml`, `megatron_actor.yaml`)
- `critic/`: Critic worker configs (`dp_critic.yaml`, `megatron_critic.yaml`)
- `ref/`: Reference policy configs (`dp_ref.yaml`, `megatron_ref.yaml`)
- `rollout/`: Rollout engine configs (`rollout.yaml`)
- `model/`: Model loading configs (`hf_model.yaml`)
- `algorithm/`: Algorithm-specific configs (`rollout_correction.yaml`)
- `engine/`: Engine backend configs (`fsdp.yaml`, `megatron.yaml`)
- `data/`: Data loading configs (`legacy_data.yaml`)

### Worker Dataclass Configs (`verl/workers/config/`)
Python `.py` files with `@dataclass` definitions:
- `actor.py`: `ActorConfig`, `PolicyLossConfig`
- `critic.py`: `CriticConfig`
- `rollout.py`: `RolloutConfig`
- `reward_model.py`: `RewardModelConfig`
