# Documentation Standards

## Module Docstrings

Place module-level docstrings immediately after the license header:

```python
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different
distributed strategies to implement PPO-like algorithms.
"""
```

Short modules may use one-line docstrings:
```python
"""The base class for Actor"""
```

## Class Docstrings

Use Google style. For worker classes, document constructor args:

```python
class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer.
            Defaults to None.
    """
```

## Config Class Docstrings

Config docstrings must include:
1. A description of the config's purpose
2. Note about BaseConfig inheritance
3. Full Args section listing every field with type and description

```python
@dataclass
class KLControlConfig(BaseConfig):
    """Configuration for KL control.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like
    interface for a dataclass config.

    Args:
        type (str): Type of KL control. Can be "fixed" or "adaptive".
        kl_coef (float): Initial coefficient for KL penalty.
        horizon (int): Horizon value for adaptive controller.
        target_kl (float): Target KL divergence for adaptive controller.
    """
```

## Method Docstrings

Use Google style with Args, Returns, and optional Raises:

```python
def compute_log_prob(self, data: DataProto) -> torch.Tensor:
    """Compute the log probability of the responses given input_ids.

    Args:
        data (DataProto): a DataProto containing keys:
            ``input_ids``: tensor of shape [batch_size, sequence_length].
            ``attention_mask``: tensor of shape [batch_size, sequence_length].
            ``responses``: tensor of shape [batch_size, response_length].

    Returns:
        torch.Tensor: the log_prob tensor
    """
```

## Function Docstrings

```python
def gather_from_labels(data, label):
    """Gather the label from data. The value in label should be [0, vocab_size)

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor): (...)

    Returns:
        tensor of gathered values
    """
```

## Inline Comments

- Use inline comments sparingly; prefer self-documenting code
- Comment tensor shapes: `# (bsz, response_length, vocab_size)`
- Comment non-obvious logic: `# prevent model thinks we are generating`
- Use `Note(username):` for implementation notes: `# Note(haibin.lin): no need to include...`
- Use `TODO:` for known improvements needed

## YAML Documentation

Every field in YAML configs must have a comment above it explaining its purpose (CI-enforced):

```yaml
# Advantage estimator type: "gae", "grpo", "reinforce_plus_plus", etc.
adv_estimator: gae

# Whether to normalize advantages by std (specific to GRPO)
norm_adv_by_std_in_grpo: True
```

## Test Docstrings

Every test function should have a docstring:

```python
def test_concat_first_worker_missing_metrics():
    """Test that metrics from other workers are preserved even when first
    worker has no metrics.

    This is a critical edge case - the old buggy implementation only checked
    data[0].meta_info and would lose all metrics if the first worker didn't
    have any.
    """
```

Test classes should include a description of what they test:

```python
class TestConfigOnCPU(unittest.TestCase):
    """Test cases for configuration utilities on CPU.

    Test Plan:
    1. Test basic OmegaConf to dataclass conversion
    2. Test nested OmegaConf to dataclass conversion
    3. Verify all configuration values are correctly converted
    """
```
