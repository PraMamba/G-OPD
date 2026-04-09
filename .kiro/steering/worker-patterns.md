# Worker Implementation Patterns

## Worker Architecture

Workers follow a Base/Implementation hierarchy:
1. Abstract base class in `base.py` with `@abstractmethod`s
2. Concrete implementations in separate files (e.g., `dp_actor.py`, `dp_critic.py`)
3. `__init__.py` exports both via `__all__`

## Base Class Pattern

```python
from abc import ABC, abstractmethod
import torch
from verl import DataProto

__all__ = ["BasePPOActor"]

class BasePPOActor(ABC):
    def __init__(self, config):
        """Args: config (DictConfig): ..."""
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute logits given a batch of data."""
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> dict:
        """Update the policy with data."""
        pass
```

Workers define 2-3 abstract methods:
- Actor: `compute_log_prob`, `update_policy`
- Critic: `compute_values`, `update_critic`
- Rollout: `resume`, `update_weights`, `release`, `generate_sequences`
- RewardManager: `__init__`, `__call__`

## Implementation Pattern

```python
class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker"""

    def __init__(self, config: ActorConfig, actor_module: nn.Module,
                 actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        # Configuration extraction from self.config
        self.use_remove_padding = self.config.get("use_remove_padding", False)
```

## Data Flow

Workers process `DataProto` objects:
- Input: `DataProto` with `.batch` (tensors), `.non_tensor_batch` (numpy/lists), `.meta_info` (metadata)
- Output: `DataProto` or tensors/dicts
- Use `data.select(batch_keys=[...])` to pick needed keys
- Use `data.split(micro_batch_size)` for micro-batching

## Logging Pattern

```python
import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
```

## GPU Memory Profiling

Use the `@GPUMemoryLogger` decorator:
```python
from verl.utils.profiler import GPUMemoryLogger

@GPUMemoryLogger(role="dp actor", logger=logger)
def compute_log_prob(self, data: DataProto, ...) -> torch.Tensor:
    ...
```

## Registry Pattern

Used for reward managers, policy losses, and advantage estimators:

```python
REGISTRY: dict[str, type] = {}

def register(name: str):
    """Decorator to register a class with a given name."""
    def decorator(cls):
        REGISTRY[name] = cls
        return cls
    return decorator

def get_cls(name: str):
    if name not in REGISTRY:
        raise ValueError(f"Unknown: {name}")
    return REGISTRY[name]
```

Usage:
```python
@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    ...
```

## Lazy Imports

Use try/except for optional dependencies:
```python
try:
    from .reward_model import RewardModelWorker
except ImportError:
    RewardModelWorker = None

__all__ = ["CriticWorker", "ActorWorker"]
if RewardModelWorker is not None:
    __all__.append("RewardModelWorker")
```

Use deferred imports inside functions for heavy dependencies:
```python
def some_method(self, ...):
    from verl.utils.model import extract_multi_modal_inputs
    multi_modal_inputs = extract_multi_modal_inputs(...)
```
