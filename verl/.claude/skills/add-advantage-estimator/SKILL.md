---
name: add-advantage-estimator
invocation: /add-advantage-estimator <name>
---

# Add Advantage Estimator

Step-by-step guide for adding a new advantage estimator to verl.

## Steps

### 1. Implement in core_algos.py
Add to `verl/trainer/ppo/core_algos.py`:

```python
@register_adv_est(name="<name>")
def compute_<name>_advantage(
    token_level_scores: torch.Tensor,     # (bsz, response_length)
    response_mask: torch.Tensor,          # (bsz, response_length)
    index: torch.Tensor | None = None,    # (bsz,) group indices for GRPO-like
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute <name> advantages.

    Args:
        token_level_scores: Per-token reward scores
        response_mask: Binary mask for valid tokens
        index: Group indices (if using group-based estimation)

    Returns:
        advantages: (bsz, response_length)
        returns: (bsz, response_length) or same as advantages
    """
    # Your implementation
    advantages = ...
    returns = advantages.clone()
    return advantages, returns
```

### 2. Add to Algorithm Config
Update `verl/trainer/config/algorithm/`:
```yaml
# New option in the config
adv_estimator: <name>
```

### 3. Test Registration
```python
def test_<name>_registered():
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn
    fn = get_adv_estimator_fn("<name>")
    assert callable(fn)
```

### 4. Add Unit Tests
Create test in `verl/tests/trainer/ppo/test_<name>_advantage.py`:

```python
import torch
from verl.trainer.ppo.core_algos import compute_<name>_advantage

def test_<name>_basic():
    scores = torch.randn(4, 128)
    mask = torch.ones(4, 128)
    adv, ret = compute_<name>_advantage(scores, mask)
    assert adv.shape == scores.shape
    assert not torch.isnan(adv).any()
```

## Reference Implementations
- `grpo`: Group-based with per-group normalization
- `rloo`: Leave-one-out baseline
- `gae`: Token-level with value function
- `reinforce_plus_plus`: Discounted returns with whitening

## Key Considerations
- **Critic required?** GAE needs a critic; GRPO/RLOO do not
- **Group-based?** If using `index`, ensure proper grouping
- **Normalization**: Use `NormConfig` parameters consistently
