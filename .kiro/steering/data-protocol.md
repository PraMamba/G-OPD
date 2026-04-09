# DataProto Protocol Patterns

## Core Data Structure

`DataProto` is the universal data transfer protocol between all workers:

```python
@dataclass
class DataProto:
    batch: TensorDict          # GPU tensors (input_ids, log_probs, etc.)
    non_tensor_batch: dict     # Non-tensor data (labels, data_source, etc.)
    meta_info: dict            # Metadata (config flags, metrics, etc.)
```

## Construction

```python
# From separate tensors and non-tensors
data = DataProto.from_dict(
    tensors={"obs": obs_tensor, "act": act_tensor},
    non_tensors={"labels": label_list},
    meta_info={"temperature": 0.7}
)

# From a mixed dict (auto-separates tensors/non-tensors)
data = DataProto.from_single_dict({"obs": tensor, "labels": list_})
```

## Key Operations

```python
# Select specific keys
data = data.select(batch_keys=["responses", "input_ids"],
                   non_tensor_batch_keys=["multi_modal_inputs"])

# Split into micro-batches
micro_batches = data.split(micro_batch_size)

# Repeat for multi-sample generation
data = data.repeat(repeat_times=n, interleave=True)

# Move to GPU
micro_batch = micro_batch.to(device_id)

# Concatenate from multiple workers
combined = DataProto.concat([data1, data2, data3])

# Index/slice
subset = data[bool_mask]
subset = data[index_array]
```

## Meta Info Conventions

- `temperature`: Model temperature (required by actors)
- `micro_batch_size`: Micro-batch size for processing
- `use_dynamic_bsz`: Whether to use dynamic batching
- `max_token_len`: Token length limit for dynamic batching
- `metrics`: List of dicts or dict of lists (auto-flattened on concat)

## Worker Interface Convention

Workers receive `DataProto`, extract needed keys, process, and return:

```python
def compute_log_prob(self, data: DataProto) -> torch.Tensor:
    micro_batch_size = data.meta_info["micro_batch_size"]
    temperature = data.meta_info["temperature"]

    data = data.select(batch_keys=["responses", "input_ids", ...])
    micro_batches = data.split(micro_batch_size)

    results = []
    for micro_batch in micro_batches:
        micro_batch = micro_batch.to(device_id)
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        with torch.no_grad():
            output = self._forward(model_inputs)
        results.append(output)

    return torch.concat(results, dim=0)
```
