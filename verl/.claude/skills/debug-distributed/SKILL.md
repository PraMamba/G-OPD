---
name: debug-distributed
invocation: /debug-distributed
---

# Debug Distributed Training

Comprehensive guide for debugging distributed training issues in verl.

## Common Issues

### 1. Hangs (Most Common)

**Symptoms**: Training freezes, no progress, no error messages.

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

### 2. Wrong Results

**Symptoms**: Loss is NaN, metrics are incorrect, gradients explode.

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

### 3. OOM (Out of Memory)

**Symptoms**: CUDA OOM errors.

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

### 4. Communication Errors

**Symptoms**: NCCL errors, "invalid device ordinal".

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

## Debugging Tools

### Ray Dashboard
```bash
# Access at http://localhost:8265
ray start --head --dashboard-host=0.0.0.0
```

### NCCL Tests
```bash
# Run NCCL bandwidth test
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g <num_gpus>
```

### Profiling
```python
# Enable PyTorch profiler (use Profiler wrapper from verl)
from verl.utils.profiler.profile import Profiler

profiler = Profiler(config)
profiler.start()
# ... training code ...
profiler.stop()
```
