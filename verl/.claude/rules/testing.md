---
name: Testing Standards
applies_to: "**/tests/**, *_test.py, test_*.py"
---

# verl Testing Standards

## Test File Naming
- `test_*_on_cpu.py`: CPU-only tests
- `test_special_*.py`: GPU tests
- No `conftest.py`: Fixtures are local to each test module

## Test Frameworks
Both pytest and unittest.TestCase are used:
- **Config tests**: Use `unittest.TestCase` with `setUp()`
- **Protocol tests**: Use plain pytest functions
- **Worker tests**: Mix of both

## Test Structure (Arrange-Act-Assert)
```python
def test_data_proto_split():
    # Arrange
    batch = TensorDict({"input_ids": torch.randn(8, 128)})
    data = DataProto(batch=batch)

    # Act
    chunks = data.split(micro_batch_size=2)

    # Assert
    assert len(chunks) == 4
```

## Pytest Markers
```python
@pytest.mark.slow           # Long-running tests
@pytest.mark.parametrize    # Parameterized tests
```

## GPU Constraints
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_fsdp_actor():
    # Test code
    torch.cuda.empty_cache()  # Cleanup
```

## Mocking Distributed
verl doesn't use `torch.distributed.fake_pg`. Instead:
- Tests run with actual Ray initialization
- Or mock at worker group level

## Fixtures
```python
@pytest.fixture
def tmp_path():  # Prefer tmp_path over tempfile
    ...

@pytest.fixture(scope="module")  # Scope appropriately
def model():
    ...
```

## Assertions
For tensors:
```python
import torch.testing

torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
```

## Edge Cases
Tests explicitly cover:
- Empty inputs
- NaN handling
- Boundary conditions
- Conflict detection

## Unittest Runner
```python
if __name__ == "__main__":
    unittest.main()
```
