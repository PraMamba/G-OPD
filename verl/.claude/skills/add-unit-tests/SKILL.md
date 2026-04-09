---
name: add-unit-tests
invocation: /add-unit-tests
---

# Add Unit Tests

Guide for adding unit tests to verl following project conventions.

## Test File Naming
- `test_<module>_on_cpu.py`: CPU-only tests
- `test_special_<module>.py`: GPU tests
- Place in `tests/` mirroring source structure

## Test Structure

### Pytest Style (Preferred for most cases)
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

### Unittest Style (For config tests)
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

## GPU Tests
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fsdp_actor():
    # Test code
    torch.cuda.empty_cache()  # Cleanup
```

## Parametrized Tests
```python
@pytest.mark.parametrize("batch_size,micro_batch_size", [
    (8, 2),
    (16, 4),
    (32, 8),
])
def test_micro_batching(batch_size, micro_batch_size):
    ...
```

## Fixtures
```python
@pytest.fixture
def sample_data():
    return DataProto(batch={"input_ids": torch.randn(4, 128)})

def test_with_fixture(sample_data):
    assert sample_data.batch["input_ids"].shape[0] == 4
```

## Assertions for Tensors
```python
import torch.testing

torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
```

## Edge Cases to Cover
- Empty inputs
- NaN/inf handling
- Boundary conditions
- Invalid configurations
