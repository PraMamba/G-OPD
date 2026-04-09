---
name: add-policy-loss
invocation: /add-policy-loss <name>
---

# Add Policy Loss Function

Step-by-step guide for adding a new policy loss to verl.

## Steps

### 1. Implement in core_algos.py
Add to `verl/trainer/ppo/core_algos.py`:

```python
@register_policy_loss(name="<name>")
def compute_<name>_policy_loss(
    old_log_prob: torch.Tensor,    # (bsz, response_length)
    log_prob: torch.Tensor,        # (bsz, response_length)
    advantages: torch.Tensor,      # (bsz, response_length)
    response_mask: torch.Tensor,   # (bsz, response_length)
    eps_clip: float = 0.2,
    **kwargs,
) -> torch.Tensor:
    """
    Compute <name> policy loss.

    Returns:
        Scalar loss tensor
    """
    # Your implementation
    ratio = torch.exp(log_prob - old_log_prob)
    loss = ...
    return loss
```

### 2. Configure in dp_actor.py
Ensure the loss is accessible via the worker config:
```yaml
actor:
  policy_loss:
    name: <name>
    eps_clip: 0.2
```

### 3. Add Unit Tests
```python
import torch
from verl.trainer.ppo.core_algos import compute_<name>_policy_loss

def test_<name>_policy_loss():
    bsz, seq_len = 4, 128
    old_lp = torch.randn(bsz, seq_len)
    new_lp = torch.randn(bsz, seq_len)
    adv = torch.randn(bsz, seq_len)
    mask = torch.ones(bsz, seq_len)

    loss = compute_<name>_policy_loss(old_lp, new_lp, adv, mask)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
```

## Reference Implementations
- `vanilla`: Standard clipped PPO with dual-clip
- `gspo`: Sequence-level importance ratio
- `gpg`: Direct `-log_prob * advantages`
