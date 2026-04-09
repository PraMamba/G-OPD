---

## Skill 4: Add Reward Function

**Invocation**: `/add-reward <name>`
**File**: `.claude/skills/add-reward/SKILL.md`

### Purpose
Step-by-step guide for adding a new reward function to verl.

### When to Use
- You want to implement custom reward logic
- You need domain-specific reward computation
- You're adding support for a new evaluation metric

### Usage Example
```
/add-reward my_custom_reward
```
or
```
"How do I add a custom reward function for code evaluation?"
```

### Steps Overview

**Step 1: Create Reward Function File**
Create `verl/utils/reward_score/my_reward.py`:
```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Compute reward score for my_reward."""
    try:
        # Your scoring logic
        score = ...
        return float(score)
    except Exception as e:
        import logging
        logger = logging.getLogger(__file__)
        logger.warning(f"Reward computation failed: {e}")
        return 0.0
```

**Step 2: Register in Reward Manager**
Configure via YAML:
```yaml
reward_fn:
  custom_reward_function: verl.utils.reward_score.my_reward.compute_score
```

**Step 3: Handle Blocking Operations**
If your reward function does blocking I/O (API calls, file I/O), ensure it's called appropriately in the reward pipeline.

**Step 4: Add Tests**
Create `verl/tests/utils/reward_score/test_my_reward.py`:
```python
import pytest
from verl.utils.reward_score.my_reward import compute_score

def test_my_reward_correct():
    score = compute_score(
        data_source="test",
        solution_str="correct answer",
        ground_truth="correct answer",
    )
    assert score > 0.5

def test_my_reward_incorrect():
    score = compute_score(
        data_source="test",
        solution_str="wrong answer",
        ground_truth="correct answer",
    )
    assert score < 0.5
```

### Key Requirements
- **Deterministic**: Same input → same output
- **Return float**: Always return a float, never None
- **Exception handling**: Catch exceptions, log, return 0.0
- **No blocking in async**: If used in async context, ensure non-blocking

### Reference Implementations
- `verl/utils/reward_score/math_dapo.py` — Math verification
- `verl/utils/reward_score/gsm8k.py` — GSM8K scoring
- `verl/utils/reward_score/prime_math/` — Multi-criteria math

---

## Skill 5: Add Unit Tests

**Invocation**: `/add-unit-tests`
**File**: `.claude/skills/add-unit-tests/SKILL.md`

### Purpose
Guide for adding unit tests to verl following project conventions.

### When to Use
- You've added new functionality that needs testing
- You want to ensure code quality and prevent regressions
- You're following TDD (Test-Driven Development)

### Usage Example
```
/add-unit-tests
```
or
```
"How do I write tests for my new advantage estimator?"
```

### Test File Naming
- `test_<module>_on_cpu.py`: CPU-only tests
- `test_special_<module>.py`: GPU tests
- Place in `tests/` mirroring source structure

### Test Structure

**Pytest Style (Preferred)**
```python
import pytest
import torch
from verl.protocol import DataProto

def test_data_proto_split():
    """Test DataProto.split() with micro-batching."""
    # Arrange
    batch = {"input_ids": torch.randn(8, 128)}
    data = DataProto(batch=batch)

    # Act
    chunks = data.split(micro_batch_size=2)

    # Assert
    assert len(chunks) == 4
    assert chunks[0].batch["input_ids"].shape[0] == 2
```

**Unittest Style (For config tests)**
```python
import unittest
from verl.workers.config import ActorConfig

class TestActorConfig(unittest.TestCase):
    def setUp(self):
        self.config = ActorConfig(model_path="test")

    def test_validation(self):
        with self.assertRaises(ValueError):
            ActorConfig(model_path="", eps_clip=-1)

if __name__ == "__main__":
    unittest.main()
```

### GPU Tests
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fsdp_actor():
    # Test code
    torch.cuda.empty_cache()  # Cleanup
```

### Parametrized Tests
```python
@pytest.mark.parametrize("batch_size,micro_batch_size", [
    (8, 2),
    (16, 4),
    (32, 8),
])
def test_micro_batching(batch_size, micro_batch_size):
    ...
```

### Fixtures
```python
@pytest.fixture
def sample_data():
    return DataProto(batch={"input_ids": torch.randn(4, 128)})

def test_with_fixture(sample_data):
    assert sample_data.batch["input_ids"].shape[0] == 4
```

### Assertions for Tensors
```python
import torch.testing

torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
```

### Edge Cases to Cover
- Empty inputs
- NaN/inf handling
- Boundary conditions
- Invalid configurations

---

## Skill 6: Debug Distributed Training

**Invocation**: `/debug-distributed`
**File**: `.claude/skills/debug-distributed/SKILL.md`

### Purpose
Comprehensive guide for debugging distributed training issues in verl.

### When to Use
- Training hangs or freezes
- Getting NCCL errors or communication failures
- Loss is NaN or metrics are incorrect
- Out of memory (OOM) errors
- Gradients are exploding or vanishing

### Usage Example
```
/debug-distributed
```
or
```
"My training is hanging, how do I debug it?"
"I'm getting NCCL errors, what should I check?"
```

### Common Issues & Solutions

#### Issue 1: Hangs (Most Common)

**Symptoms**: Training freezes, no progress, no error messages

**Causes**:
- Mismatched collectives (all-reduce called by subset of ranks)
- Deadlock in Ray actor communication
- NCCL timeout

**Debug Steps**:
```bash
# Enable detailed logging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0

# Run with timeout
timeout 300 python -m verl.trainer.main_ppo ...
```

**Check**:
- Are all ranks calling the same collective?
- Is there a barrier or all-reduce in a conditional branch?
- Check Ray dashboard for stuck actors

#### Issue 2: Wrong Results

**Symptoms**: Loss is NaN, metrics are incorrect, gradients explode

**Causes**:
- Incorrect ReduceOp (SUM vs MEAN)
- Unsharded tensors
- Wrong process group

**Debug Steps**:
```python
# Add logging in worker code
logger.info(f"Rank {rank}: tensor shape {tensor.shape}, mean {tensor.mean()}")

# Check gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        logger.info(f"{name}: grad norm {param.grad.norm()}")
```

#### Issue 3: OOM (Out of Memory)

**Symptoms**: CUDA OOM errors

**Causes**:
- Unsharded tensors accumulating
- Gradient accumulation not clearing
- vLLM/SGLang not releasing memory

**Debug Steps**:
```python
# Add memory profiling
from verl.utils.profiler import GPUMemoryLogger

@GPUMemoryLogger
def my_function():
    ...

# Manual cleanup
torch.cuda.empty_cache()
```

**Check**:
- Is `ShardingManager.sync_weights_from_rollout()` called?
- Are gradients cleared between micro-batches?
- Is FSDP wrapping applied correctly?

#### Issue 4: Communication Errors

**Symptoms**: NCCL errors, "invalid device ordinal"

**Causes**:
- Wrong device placement
- Process group mismatch
- NCCL version incompatibility

**Debug Steps**:
```bash
# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Check device visibility
echo $CUDA_VISIBLE_DEVICES

# Test NCCL
python -c "import torch; torch.distributed.init_process_group('nccl'); print('OK')"
```

### Debugging Tools

**Ray Dashboard**
```bash
# Access at http://localhost:8265
ray start --head --dashboard-host=0.0.0.0
```

**NCCL Tests**
```bash
# Run NCCL bandwidth test
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g <num_gpus>
```

**Profiling**
```python
# Enable PyTorch profiler
from verl.utils.profiler.profile import Profiler

profiler = Profiler(config)
profiler.start()
# ... training code ...
profiler.stop()
```

---

## Skills Usage Tips

### 1. Follow the Steps in Order
Skills provide sequential steps. Follow them in order for best results.

### 2. Adapt to Your Use Case
Skills provide templates. Adapt the code to your specific requirements.

### 3. Check Reference Implementations
Each skill points to existing code examples. Study them before implementing.

### 4. Test Thoroughly
Always add tests as specified in the skill. This prevents regressions.

### 5. Use Code Verifier After
After following a skill, the Code Verifier agent will automatically check your work.

---

## Skill Workflow Example

```
1. User invokes skill: /add-reward my_math_reward

2. Skill provides step-by-step guide

3. User implements following the steps:
   - Creates verl/utils/reward_score/my_math_reward.py
   - Implements compute_score() function
   - Configures in YAML
   - Adds tests

4. Code Verifier automatically runs:
   - Checks code style (ruff)
   - Runs tests
   - Reports any issues

5. User iterates if needed
```
