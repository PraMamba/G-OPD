# Configuration Patterns

## Dual Config System

The codebase uses two configuration layers:
1. **YAML configs** (Hydra) for user-facing configuration
2. **Python dataclasses** (`BaseConfig` subclasses) for runtime type safety

## BaseConfig Dataclass Pattern

All config dataclasses inherit from `BaseConfig` which provides:
- Dict-like access (`config["key"]`, `config.get("key", default)`)
- Iteration, `len()` support (Mapping ABC)
- Field immutability by default (frozen fields)
- Mutable fields via `_mutable_fields` class variable

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from verl.base_config import BaseConfig

__all__ = ["PolicyLossConfig", "ActorConfig"]

@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like
    interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode.
        clip_cov_ratio (float): Ratio of tokens to be clipped.
    """
    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    only_reverse_kl_advantages: bool = False
    lambda_vals: float = 1.0
```

## Config Hierarchy

Configs compose via nesting:
```python
@dataclass
class ActorConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {"ppo_mini_batch_size", ...}

    strategy: str = MISSING  # Required field
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        # validation logic...

    def validate(self, n_gpus: int, train_batch_size: int, ...):
        """Validate actor configuration with runtime parameters."""
        ...
```

Specializations use inheritance:
```python
@dataclass
class FSDPActorConfig(ActorConfig):
    strategy: str = "fsdp"
    grad_clip: float = 1.0
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
```

## Config with Factory Methods

Complex configs can provide named factory methods:
```python
@dataclass
class RolloutCorrectionConfig(BaseConfig):
    rollout_is: Optional[str] = "sequence"
    rollout_is_threshold: float = 2.0

    @classmethod
    def decoupled_token_is(cls, threshold: float = 2.0) -> "RolloutCorrectionConfig":
        """Decoupled Mode with Token-level Importance Sampling."""
        return cls(rollout_is="token", rollout_is_threshold=threshold)

    @classmethod
    def disabled(cls) -> "RolloutCorrectionConfig":
        """Disabled - Metrics Only Mode."""
        return cls(rollout_is=None, rollout_rs=None)
```

## YAML Config Conventions

YAML files in `verl/trainer/config/` follow strict format (CI-enforced):
1. Comments must appear **above** each field (not inline)
2. Blank line between each field
3. Inline comments (after a field on the same line) are not allowed
4. Indentation level is respected for nested fields
5. Include `_target_` for dataclass instantiation

```yaml
# Format checks enforced on CI:
# 1. Comments must appear above each field.
# 2. There must be a blank line between each field.
# 3. Inline comments (after a field on the same line) are not allowed.

# Target class for this configuration
_target_: verl.workers.config.FSDPActorConfig

# Training strategy
strategy: fsdp

# Gradient clipping for actor updates
grad_clip: 1.0
```

## YAML-to-Dataclass Conversion

Use `omega_conf_to_dataclass` to convert Hydra configs to typed dataclasses:
```python
from verl.utils.config import omega_conf_to_dataclass

config = omega_conf_to_dataclass(cfg)  # Auto-detects class from _target_
config = omega_conf_to_dataclass(cfg, dataclass_type=HFModelConfig)  # Explicit type
```

## Package Organization

- `verl/trainer/config/` - Trainer/algorithm-level configs (AlgoConfig, CheckpointConfig)
- `verl/workers/config/` - Worker-level configs (ActorConfig, CriticConfig, RolloutConfig)
- Each config module has `__init__.py` with `__all__` aggregating sub-module exports
