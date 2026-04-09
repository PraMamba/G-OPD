# Testing Patterns

## Test Organization

Tests mirror the package structure under `verl/tests/`:
- `tests/workers/actor/` tests `verl/workers/actor/`
- `tests/workers/config/` tests `verl/workers/config/`
- `tests/utils/` tests `verl/utils/`
- Root-level tests (`tests/test_protocol_on_cpu.py`) test core modules

## Test File Naming

Use the suffix convention to indicate hardware requirements:
- `_on_cpu.py` - CPU-only tests (no GPU needed), can run in CI
- `test_special_*.py` - Tests requiring special hardware (GPU, multi-GPU)
- No suffix - Standard tests

## Test Frameworks

The codebase uses both frameworks interchangeably:
- **pytest**: Preferred for new tests, used with functions and fixtures
- **unittest.TestCase**: Also used, especially for config tests

## Pytest Patterns

### Fixtures

```python
@pytest.fixture
def base_config_mock():
    """Fixture to create a mock BaseConfig instance with test attributes."""
    mock_config = BaseConfig()
    mock_config.test_attr = "test_value"
    return mock_config
```

### Skip conditions

```python
@pytest.mark.skipif(
    parse_version(tensordict.__version__) < parse_version("0.10"),
    reason="requires at least tensordict 0.10"
)
def test_to_tensordict():
    ...
```

### Exception testing

```python
with pytest.raises(AssertionError, match="Conflicting values for meta_info key"):
    DataProto.concat([data1, data2])
```

## unittest Patterns

```python
class TestActorConfig(unittest.TestCase):
    """Test the ActorConfig dataclass and its variants."""

    def test_config_inheritance(self):
        """Test that the inheritance hierarchy works correctly."""
        ...

    def test_frozen_fields_modification_raises_exception(self):
        """Test that modifying frozen fields raises an exception."""
        with self.assertRaises(AttributeError):
            config.strategy = "megatron"

if __name__ == "__main__":
    unittest.main()
```

## Test Style

- Each test function has a docstring explaining what it tests
- Tests cover happy path, exception paths, and edge cases
- Use descriptive test names: `test_<what>_<condition>` (e.g., `test_getitem_nonexistent_attribute`)
- Use `torch.testing.assert_close` for tensor comparisons
- Use `np.array_equal` for numpy comparisons
- No conftest.py files - fixtures are defined locally in test files

## Running Tests

```bash
cd verl/
pytest tests/                             # Run all tests
pytest tests/test_protocol_on_cpu.py      # Run specific test file
ruff check .                              # Lint check (line-length=120)
```
