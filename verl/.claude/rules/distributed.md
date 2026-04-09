---
name: Distributed Training Standards
applies_to: "verl/workers/**, verl/trainer/**, verl/single_controller/**"
---

# verl Distributed Training Standards

## Ray-Based Architecture
verl uses Ray for distributed coordination, not torch.distributed directly in most cases.

## Worker Communication
All inter-worker communication uses `DataProto` objects dispatched via decorators:
```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_log_prob(self, data: DataProto) -> DataProto:
    ...
```

## Dispatch Modes
- `DP_COMPUTE_PROTO`: Splits DataProto across DP ranks, collects results
- `DP_COMPUTE_METRIC`: For metric aggregation
- `RANK_ZERO`: Execute only on rank 0
- `ONE_TO_ALL`: Broadcast from driver to all workers

## Process Groups
- Pass process groups explicitly, never create global PG at module level
- Use `get_device_id()`, `get_device_name()`, `get_torch_device()` from `verl.utils.device`

## FSDP Patterns
```python
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

# Apply FSDP wrapping
model = FSDP(
    model,
    auto_wrap_policy=get_fsdp_wrap_policy(model),
    ...
)
```

## Common Pitfalls
- **Hangs**: Mismatched collectives (all-reduce called by subset of ranks)
- **Wrong results**: Incorrect ReduceOp (SUM vs MEAN)
- **OOM**: Unsharded tensors accumulating

## Debugging
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0
```

## Tensor Parallelism
verl supports TP via vLLM/SGLang rollout engines. Worker code should be TP-agnostic.

## Sequence Balancing
Use `get_seqlen_balanced_partitions()` to balance workload across DP ranks:
```python
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions

partitions = get_seqlen_balanced_partitions(
    seq_lens, n_partitions=world_size
)
```
