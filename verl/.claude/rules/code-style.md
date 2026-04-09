---
name: Code Style Standards
applies_to: "**/*.py"
---

# verl Code Style Standards

## License Headers
Every `.py` file starts with Apache 2.0 header. Multi-contributor files list all copyrights.

## Import Order
1. Standard library
2. Third-party (torch, numpy, ray, transformers)
3. verl package imports

Enforced by ruff+isort with `known-first-party = ["verl"]`.

## Modern Python Syntax
- Use `dict[str, Any]` not `Dict[str, Any]`
- Use `X | Y` for unions not `Union[X, Y]`
- Use `type[X]` not `Type[X]`
- pyupgrade rules enforced

## Line Length
120 characters (not 88). Configured in ruff.

## Public API Control
All `__init__.py` files and key modules define `__all__` to control public API surface.

## Logging Pattern
```python
import logging
import os

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
```

## Aliased Imports
Project-wide convention:
```python
import verl.utils.torch_functional as verl_F
```

## Design Patterns
- **Composition over inheritance**: Shallow hierarchies (≤2 levels)
- **Registry pattern**: Used for reward managers, policy losses, advantage estimators
  ```python
  @register("name")
  def my_function(...):
      ...
  ```

## Naming Conventions
- Configs: `XxxConfig` (dataclasses)
- Workers: `XxxWorker`, `XxxActor`, `XxxCritic`
- Managers: `XxxManager`
- Functions: `snake_case`

## Tensor Conventions
Default shape: `[batch, seq_len, hidden]` or document clearly with inline comments:
```python
logits = model(input_ids)  # (bsz, response_length, vocab_size)
```

## Performance Patterns
- Avoid GPU-CPU sync: `.item()`, `.tolist()`, `.cpu()` in hot paths
- Prefer batch operations over loops
- Use `@GPUMemoryLogger` decorator for memory profiling

## Lazy Imports
Heavy or optional dependencies imported inside methods, not at module top:
```python
def compute_reward():
    try:
        from math_verify import verify_answer
    except ImportError:
        raise ImportError("Install math-verify: pip install math-verify")
```

## DataProto Usage
All worker communication uses `DataProto`:
```python
# Extract keys
data.select(["input_ids", "attention_mask"])

# Split for micro-batching
for micro_batch in data.split(micro_batch_size):
    process(micro_batch)
```
