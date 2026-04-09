---
name: add-dataset
invocation: /add-dataset <name>
---

# Add Dataset

Step-by-step guide for adding a new dataset to verl.

## Steps

### 1. Create Dataset File
Create `verl/utils/dataset/<name>.py`:

```python
"""
<Name> dataset loader for verl.
"""
from datasets import load_dataset

def get_<name>_dataset(split: str = "train"):
    """
    Load <name> dataset.

    Args:
        split: Dataset split (train/test/validation)

    Returns:
        Dataset with required fields
    """
    dataset = load_dataset("<hf_path>", split=split)

    # Transform to verl format
    def transform(example):
        return {
            "data_source": "<name>",
            "prompt": example["question"],  # or construct from messages
            "ability": "math",  # or "code", "qa", etc.
            "reward_model": {
                "ground_truth": example["answer"],
                "style": "rule",  # or "model"
            },
            # Optional fields
            "extra_info": {},
        }

    return dataset.map(transform)
```

### 2. Register Dataset
Add to your training script or config:
```python
from verl.utils.dataset.<name> import get_<name>_dataset

train_dataset = get_<name>_dataset(split="train")
```

Or via Hydra config:
```yaml
data:
  train_files: null
  train_dataset_builder:
    _target_: verl.utils.dataset.<name>.get_<name>_dataset
    split: train
```

### 3. Required Fields
**Minimum required**:
- `data_source`: Dataset identifier (str)
- `prompt`: Input prompt (str) or `messages` (list of dicts)
- `ability`: Task type (str)
- `reward_model.ground_truth`: Expected answer (str)
- `reward_model.style`: "rule" or "model"

**Optional**:
- `extra_info`: Additional metadata (dict)
- `answer`: For validation/logging

### 4. Add Tests
Create `verl/tests/utils/dataset/test_<name>.py`:

```python
from verl.utils.dataset.<name> import get_<name>_dataset

def test_<name>_dataset_load():
    dataset = get_<name>_dataset(split="train")
    assert len(dataset) > 0

    sample = dataset[0]
    assert "data_source" in sample
    assert "prompt" in sample or "messages" in sample
    assert "reward_model" in sample
```

## Reference Implementations
Check existing datasets in `verl/utils/dataset/` for examples.
