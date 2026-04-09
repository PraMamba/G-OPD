---
name: add-reward
invocation: /add-reward <name>
---

# Add Reward Function

Step-by-step guide for adding a new reward function to verl.

## Steps

### 1. Create Reward Function File
Create `verl/utils/reward_score/<name>.py`:

```python
"""
<Name> reward function for verl.
"""

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """
    Compute reward score for <name>.

    Args:
        data_source: Dataset identifier
        solution_str: Model-generated solution
        ground_truth: Expected answer
        extra_info: Optional metadata

    Returns:
        Reward score (float)
    """
    # Implementation here
    try:
        # Your scoring logic
        score = ...
        return float(score)
    except Exception as e:
        # Log error and return 0.0
        import logging
        logger = logging.getLogger(__file__)
        logger.warning(f"Reward computation failed: {e}")
        return 0.0
```

### 2. Register in Reward Manager
If using custom reward function, configure via YAML:
```yaml
reward_fn:
  custom_reward_function: verl.utils.reward_score.<name>.compute_score
```

Or create a custom reward manager in `verl/workers/reward_manager/`.

### 3. Handle Blocking Operations
If your reward function does blocking I/O (API calls, file I/O), ensure it's called appropriately in the reward pipeline.

### 4. Add Tests
Create `verl/tests/utils/reward_score/test_<name>.py`:

```python
import pytest
from verl.utils.reward_score.<name> import compute_score

def test_<name>_correct():
    score = compute_score(
        data_source="test",
        solution_str="correct answer",
        ground_truth="correct answer",
    )
    assert score > 0.5

def test_<name>_incorrect():
    score = compute_score(
        data_source="test",
        solution_str="wrong answer",
        ground_truth="correct answer",
    )
    assert score < 0.5
```

## Key Requirements
- **Deterministic**: Same input → same output
- **Return float**: Always return a float, never None
- **Exception handling**: Catch exceptions, log, return 0.0
- **No blocking in async**: If used in async context, ensure non-blocking

## Reference Implementations
- `verl/utils/reward_score/math_dapo.py` — Math verification
- `verl/utils/reward_score/gsm8k.py` — GSM8K scoring
- `verl/utils/reward_score/prime_math/` — Multi-criteria math
