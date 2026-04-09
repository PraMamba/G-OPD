# verl Claude Skills Guide

This guide explains how to use the skills in the verl `.claude` configuration. Skills are step-by-step guides for common development tasks.

## Overview

The verl `.claude` configuration includes 6 skills that provide structured workflows for common tasks:

1. **add-advantage-estimator** - Add new advantage estimators to the registry
2. **add-policy-loss** - Add new policy loss functions
3. **add-dataset** - Add new dataset loaders
4. **add-reward** - Add new reward functions
5. **add-unit-tests** - Add unit tests following verl conventions
6. **debug-distributed** - Debug distributed training issues

## How to Use Skills

Skills are invoked using the `/skill-name` syntax or by asking about the task directly.

---

## Skill 1: Add Advantage Estimator

**Invocation**: `/add-advantage-estimator <name>`
**File**: `.claude/skills/add-advantage-estimator/SKILL.md`

### Purpose
Step-by-step guide for adding a new advantage estimator to verl's extensible registry.

### When to Use
- You want to implement a new advantage estimation algorithm
- You're experimenting with custom advantage computation
- You need to extend verl's algorithm capabilities

### Usage Example
```
/add-advantage-estimator my_custom_estimator
```
or
```
"How do I add a new advantage estimator called 'my_custom_estimator'?"
```

### Steps Overview

**Step 1: Implement in core_algos.py**
Add your function to `verl/trainer/ppo/core_algos.py`:
```python
@register_adv_est(name="my_custom")
def compute_my_custom_advantage(
    token_level_scores: torch.Tensor,     # (bsz, response_length)
    response_mask: torch.Tensor,          # (bsz, response_length)
    index: torch.Tensor | None = None,    # (bsz,) group indices
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute my_custom advantages."""
    # Your implementation
    advantages = ...
    returns = advantages.clone()
    return advantages, returns
```

**Step 2: Add to Algorithm Config**
Update `verl/trainer/config/algorithm/`:
```yaml
adv_estimator: my_custom
```

**Step 3: Test Registration**
```python
def test_my_custom_registered():
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    fn = get_adv_estimator_fn("my_custom")
    assert callable(fn)
```

**Step 4: Add Unit Tests**
Create `verl/tests/trainer/ppo/test_my_custom_advantage.py`:
```python
import torch
from verl.trainer.ppo.core_algos import compute_my_custom_advantage

def test_my_custom_basic():
    scores = torch.randn(4, 128)
    mask = torch.ones(4, 128)
    adv, ret = compute_my_custom_advantage(scores, mask)
    assert adv.shape == scores.shape
    assert not torch.isnan(adv).any()
```

### Key Considerations
- **Critic required?** GAE needs a critic; GRPO/RLOO do not
- **Group-based?** If using `index`, ensure proper grouping
- **Normalization**: Use `NormConfig` parameters consistently

### Reference Implementations
- `grpo`: Group-based with per-group normalization
- `rloo`: Leave-one-out baseline
- `gae`: Token-level with value function
- `reinforce_plus_plus`: Discounted returns with whitening

---

## Skill 2: Add Policy Loss

**Invocation**: `/add-policy-loss <name>`
**File**: `.claude/skills/add-policy-loss/SKILL.md`

### Purpose
Step-by-step guide for adding a new policy loss function to verl's registry.

### When to Use
- You want to implement a custom policy gradient loss
- You're experimenting with different clipping strategies
- You need to extend verl's policy optimization methods

### Usage Example
```
/add-policy-loss my_custom_loss
```
or
```
"How do I add a new policy loss function?"
```

### Steps Overview

**Step 1: Implement in core_algos.py**
```python
@register_policy_loss(name="my_custom")
def compute_my_custom_policy_loss(
    old_log_prob: torch.Tensor,    # (bsz, response_length)
    log_prob: torch.Tensor,        # (bsz, response_length)
    advantages: torch.Tensor,      # (bsz, response_length)
    response_mask: torch.Tensor,   # (bsz, response_length)
    eps_clip: float = 0.2,
    **kwargs,
) -> torch.Tensor:
    """Compute my_custom policy loss."""
    ratio = torch.exp(log_prob - old_log_prob)
    loss = ...  # Your implementation
    return loss
```

**Step 2: Configure in dp_actor.py**
```yaml
actor:
  policy_loss:
    name: my_custom
    eps_clip: 0.2
```

**Step 3: Add Unit Tests**
```python
import torch
from verl.trainer.ppo.core_algos import compute_my_custom_policy_loss

def test_my_custom_policy_loss():
    bsz, seq_len = 4, 128
    old_lp = torch.randn(bsz, seq_len)
    new_lp = torch.randn(bsz, seq_len)
    adv = torch.randn(bsz, seq_len)
    mask = torch.ones(bsz, seq_len)

    loss = compute_my_custom_policy_loss(old_lp, new_lp, adv, mask)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
```

### Reference Implementations
- `vanilla`: Standard clipped PPO with dual-clip
- `gspo`: Sequence-level importance ratio
- `gpg`: Direct `-log_prob * advantages`

---

## Skill 3: Add Dataset

**Invocation**: `/add-dataset <name>`
**File**: `.claude/skills/add-dataset/SKILL.md`

### Purpose
Step-by-step guide for adding a new dataset to verl.

### When to Use
- You want to train on a custom dataset
- You need to load data from HuggingFace or local files
- You're adding support for a new task domain

### Usage Example
```
/add-dataset my_math_dataset
```
or
```
"How do I add a new dataset for math problems?"
```

### Steps Overview

**Step 1: Create Dataset File**
Create `verl/utils/dataset/my_dataset.py`:
```python
from datasets import load_dataset

def get_my_dataset(split: str = "train"):
    """Load my_dataset."""
    dataset = load_dataset("path/to/dataset", split=split)

    def transform(example):
        return {
            "data_source": "my_dataset",
            "prompt": example["question"],
            "ability": "math",  # or "code", "qa"
            "reward_model": {
                "ground_truth": example["answer"],
                "style": "rule",  # or "model"
            },
            "extra_info": {},
        }

    return dataset.map(transform)
```

**Step 2: Register Dataset**
Via Hydra config:
```yaml
data:
  train_dataset_builder:
    _target_: verl.utils.dataset.my_dataset.get_my_dataset
    split: train
```

**Step 3: Required Fields**
- `data_source`: Dataset identifier (str)
- `prompt`: Input prompt (str) or `messages` (list)
- `ability`: Task type (str)
- `reward_model.ground_truth`: Expected answer (str)
- `reward_model.style`: "rule" or "model"

**Step 4: Add Tests**
```python
from verl.utils.dataset.my_dataset import get_my_dataset

def test_my_dataset_load():
    dataset = get_my_dataset(split="train")
    assert len(dataset) > 0
    sample = dataset[0]
    assert "data_source" in sample
    assert "reward_model" in sample
```
