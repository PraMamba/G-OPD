# Multi-Teacher Distillation: Code Review

Detailed code-level review of the ExOPD multi-teacher distillation implementation.

## Table of Contents

- [1. Core Multi-Teacher Logic](#1-core-multi-teacher-logic)
- [2. Training Loop Orchestration](#2-training-loop-orchestration)
- [3. FSDP Worker Model Management](#3-fsdp-worker-model-management)
- [4. Data Pipeline](#4-data-pipeline)
- [5. Re-tokenization](#5-re-tokenization)
- [6. Advantage Overwrite Pattern](#6-advantage-overwrite-pattern)
- [7. Issues Summary](#7-issues-summary)
- [8. Positive Observations](#8-positive-observations)

---

## 1. Core Multi-Teacher Logic

**File**: `verl/verl/workers/actor/dp_actor.py` (lines 494-533)

This is the heart of multi-teacher distillation — per-sample reverse KL computation routed by teacher type.

```python
if self.config.policy_loss.multi_teacher_distill:
    #### multi-teacher distillation ####
    if "opd_teacher" in model_inputs:
        opd_teacher = model_inputs["opd_teacher"]
        batch_size = old_log_prob.shape[0]
        reverse_kl = torch.zeros_like(old_log_prob)

        for i in range(batch_size):
            teacher_type = opd_teacher[i]
            if teacher_type == "math":
                if lambda_vals == 1.0:
                    reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                else:
                    reverse_kl[i] = (old_log_prob[i] - model_inputs["base_log_prob"][i]
                        - (model_inputs["ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals)
            elif teacher_type == "code":
                if lambda_vals == 1.0:
                    reverse_kl[i] = old_log_prob[i] - model_inputs["base_ref_log_prob"][i]
                else:
                    reverse_kl[i] = (old_log_prob[i] - model_inputs["base_log_prob"][i]
                        - (model_inputs["base_ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals)
            else:
                reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]

    advantages = (- (reverse_kl))
```

### Issues Found

**[Critical] Python loop over batch dimension (line 502)**

The `for i in range(batch_size)` loop iterates over individual samples in Python. With batch sizes of 1024, this is unnecessarily slow. A vectorized approach using boolean masks would eliminate the loop entirely:

```python
# Recommended vectorized approach
math_mask = torch.tensor([t == "math" for t in opd_teacher]).unsqueeze(-1).float()
code_mask = torch.tensor([t == "code" for t in opd_teacher]).unsqueeze(-1).float()
other_mask = ~(math_mask.bool() | code_mask.bool()).float()

if lambda_vals == 1.0:
    math_kl = old_log_prob - model_inputs["ref_log_prob"]
    code_kl = old_log_prob - model_inputs["base_ref_log_prob"]
else:
    base_diff = old_log_prob - model_inputs["base_log_prob"]
    math_kl = base_diff - (model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]) * lambda_vals
    code_kl = base_diff - (model_inputs["base_ref_log_prob"] - model_inputs["base_log_prob"]) * lambda_vals

other_kl = old_log_prob - model_inputs["ref_log_prob"]
reverse_kl = math_mask * math_kl + code_mask * code_kl + other_mask * other_kl
```

The code itself acknowledges this with `# TODO: need to improve the logic here` (line 504).

**[High] Hardcoded teacher type strings (lines 505, 510)**

String comparisons `"math"` and `"code"` are hardcoded with no extensibility mechanism. Adding a third teacher requires modifying this core training loop.

**[Medium] Floating-point equality check (lines 506, 511)**

`if lambda_vals == 1.0:` uses exact float comparison. While typically safe for config values, `math.isclose(lambda_vals, 1.0)` would be more robust.

**[Medium] Silent fallback for unknown teacher types (line 516)**

The `else` branch silently defaults to using `ref_log_prob` (math teacher) for unrecognized teacher types. Should at minimum log a warning.

**[Medium] Same `lambda_vals` for all teachers (line 492)**

A single scaling factor is applied uniformly. Different teachers may have different capability gaps relative to the base, warranting per-teacher scaling.

---

## 2. Training Loop Orchestration

**File**: `verl/verl/trainer/ppo/ray_trainer.py` (lines 1200-1233)

### Sequential Teacher Log-Prob Computation

```python
if self.use_base_models:
    # Step A: Code teacher log-probs
    base_ref_log_prob = self.ref_policy_wg.compute_base_ref_log_prob(batch)
    batch = batch.union(base_ref_log_prob)

    # Step B: Temporarily remove ref_input_ids
    ref_input_tensors = {}
    if "ref_input_ids" in batch.batch:
        ref_input_tensors["ref_input_ids"] = batch.batch.pop("ref_input_ids")
    if "ref_attention_mask" in batch.batch:
        ref_input_tensors["ref_attention_mask"] = batch.batch.pop("ref_attention_mask")
    if "ref_position_ids" in batch.batch:
        ref_input_tensors["ref_position_ids"] = batch.batch.pop("ref_position_ids")

    # Step C: Student base log-probs (using actor's input_ids)
    base_log_prob = self.actor_rollout_wg.compute_base_log_prob(batch)
    batch = batch.union(base_log_prob)

    # Restore ref_input_ids
    for key, tensor in ref_input_tensors.items():
        batch.batch[key] = tensor
```

### Issues Found

**[High] Fragile pop/restore pattern (lines 1216-1229)**

The temporary removal and restoration of `ref_input_ids` is a workaround to reuse `compute_log_prob` with the correct input tensors. If an exception occurs between pop and restore, the batch is left in an inconsistent state. Should use try/finally:

```python
try:
    ref_input_tensors = {}
    for key in ["ref_input_ids", "ref_attention_mask", "ref_position_ids"]:
        if key in batch.batch:
            ref_input_tensors[key] = batch.batch.pop(key)
    base_log_prob = self.actor_rollout_wg.compute_base_log_prob(batch)
    batch = batch.union(base_log_prob)
finally:
    for key, tensor in ref_input_tensors.items():
        batch.batch[key] = tensor
```

**[Low] Print statement on every step (line 1231)**

```python
print(f"Computed base log probs for corrected reward: ...")
```

Executes on every training step. Should be `logger.debug()` or gated to first step only.

### Constructor Setup (`ray_trainer.py:332-341`)

```python
self.base_model_path = config.actor_rollout_ref.model.get("base_model_path", None)
self.ref_base_model_path = config.actor_rollout_ref.ref.get("model", None)
if self.ref_base_model_path is not None:
    self.ref_base_model_path = self.ref_base_model_path.get("base_model_path", None)
self.use_base_models = self.base_model_path is not None and self.ref_base_model_path is not None
```

The `use_base_models` flag gates all multi-teacher log-prob computation. It is `True` when both `base_model_path` (student base) and `ref.model.base_model_path` (code teacher) are configured.

---

## 3. FSDP Worker Model Management

**File**: `verl/verl/workers/fsdp_workers.py` (lines 840-900, 1097-1172)

### Model Initialization

Base models reuse existing infrastructure (`_build_model_optimizer`) with no optimizer (`optim_config=None`):

```python
# Actor base model (lines 844-869)
base_model_path = self.config.model.get("base_model_path", None)
if base_model_path is not None and self._is_actor:
    self.base_module_fsdp = self._build_model_optimizer(
        model_path=local_base_path,
        fsdp_config=...,
        optim_config=None,
        role="ref",  # CPU offload
    )[0]
```

### Log-Prob Computation Methods

**[Medium] DRY violation**: `compute_base_log_prob` and `compute_base_ref_log_prob` (lines 1097-1172) are near-identical, differing only in:
1. Which model: `self.base_policy` vs `self.base_ref_policy`
2. Output key: `"base_log_prob"` vs `"base_ref_log_prob"`
3. Flag: `self._has_base_model` vs `self._has_base_ref_model`

A single parameterized helper would eliminate duplication.

---

## 4. Data Pipeline

**File**: `verl/verl/utils/dataset/rl_dataset.py` (lines 454-455)

```python
if "opd_teacher" in row_dict.get("extra_info", {}):
    row_dict["opd_teacher"] = row_dict.get("extra_info", {}).get("opd_teacher")
```

Clean and minimal. The `opd_teacher` field is extracted from the parquet data's `extra_info` column and promoted to a top-level key. It flows through the DataProto pipeline as a `non_tensor_batch` field.

In `dp_actor.py:420-422`, it is preserved during data selection:

```python
if "opd_teacher" in data.non_tensor_batch.keys():
    non_tensor_select_keys.append("opd_teacher")
```

---

## 5. Re-tokenization

**File**: `verl/verl/trainer/ppo/ref_input_utils.py` (line 36+)

Handles cases where the teacher model uses a different tokenizer than the student. Extracts `raw_prompt`, applies the ref tokenizer's chat template, left-pads, and concatenates with response tokens.

**[Medium] Only supports single ref tokenizer**: `prepare_ref_model_inputs` accepts a single `ref_tokenizer`. In multi-teacher mode, both teachers may need different tokenizers. Currently works only when all models share the same tokenizer family.

---

## 6. Advantage Overwrite Pattern

**Files**: `verl/verl/trainer/ppo/core_algos.py` (lines 264-328) + `verl/verl/workers/actor/dp_actor.py` (line 533)

When `only_reverse_kl_advantages=True`, the GRPO advantage estimator runs (computing group-normalized reward advantages) but its output is immediately **overwritten**:

```python
# In dp_actor.py update_policy():
advantages = (- (reverse_kl))  # Overwrites GRPO advantages
```

**[High]**: The GRPO computation in `core_algos.py:264-328` runs unnecessarily, consuming compute. This could be short-circuited by skipping `compute_advantage` when `only_reverse_kl_advantages=True`.

---

## 7. Issues Summary

### Critical

| ID | File | Lines | Description |
|----|------|-------|-------------|
| C1 | `dp_actor.py` | 502-516 | Python loop over batch dimension; should be vectorized with boolean masks |

### High

| ID | File | Lines | Description |
|----|------|-------|-------------|
| H1 | `dp_actor.py` | 505, 510 | Hardcoded teacher type strings with no extensibility |
| H2 | `ray_trainer.py` | 1216-1229 | Fragile pop/restore pattern without try/finally |
| H3 | `core_algos.py` + `dp_actor.py` | 264-328, 533 | GRPO advantages computed then overwritten |

### Medium

| ID | File | Lines | Description |
|----|------|-------|-------------|
| M1 | `dp_actor.py` | 506, 511 | Float equality check `lambda_vals == 1.0` |
| M2 | `dp_actor.py` | 516 | Silent fallback for unknown teacher types |
| M3 | `ray_trainer.py` | 332-336 | `ref.model.base_model_path` overloaded semantically |
| M4 | `ref_input_utils.py` | 36 | Re-tokenization only supports single ref tokenizer |
| M5 | `dp_actor.py` | 492 | Same `lambda_vals` for all teachers |
| M6 | Multi-teacher script | 47-48 | `use_kl_loss=True` with `kl_loss_coef=0` wastes compute |
| M7 | `fsdp_workers.py` | 1097-1172 | Near-identical `compute_base_log_prob` / `compute_base_ref_log_prob` |

### Low

| ID | File | Lines | Description |
|----|------|-------|-------------|
| L1 | `config/actor.py` | 56 | `lambda_vals` (plural) holds single float |
| L2 | `config/actor.py` | 56 | No range validation for `lambda_vals` |
| L3 | `ray_trainer.py` | 1231 | Print statement on every training step |

---

## 8. Positive Observations

1. **Rollout Correction Module** (`rollout_corr_helper.py`): Exceptionally well-documented with mathematical references, safety bounds (`SAFETY_BOUND = 20.0`), ESS metrics, and clean API design. A model for numerical RL code.

2. **Type-safe Config System**: `BaseConfig` dataclass inheritance with factory methods (e.g., `RolloutCorrectionConfig.decoupled_token_is()`) provides clean preset patterns.

3. **Clean Data Pipeline**: The `opd_teacher` propagation from parquet → `rl_dataset.py` → `non_tensor_batch` → `dp_actor.py` is minimal and well-integrated.

4. **FSDP Worker Reuse**: Base model initialization leverages existing `_build_model_optimizer` infrastructure without duplicating model loading logic.

5. **Single/Multi Toggle**: The `multi_teacher_distill` boolean provides a clean switch between modes while keeping the single-teacher path simple and efficient.
