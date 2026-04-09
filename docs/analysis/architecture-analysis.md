# Multi-Teacher Distillation: Architecture Analysis

Deep architectural analysis of the ExOPD multi-teacher distillation system.

## Table of Contents

- [1. System Architecture](#1-system-architecture)
- [2. Component Interaction Map](#2-component-interaction-map)
- [3. FSDP Worker Model Colocation](#3-fsdp-worker-model-colocation)
- [4. Rollout Correction Subsystem](#4-rollout-correction-subsystem)
- [5. Configuration Architecture](#5-configuration-architecture)
- [6. Scalability Analysis](#6-scalability-analysis)
- [7. Key Design Decisions & Trade-offs](#7-key-design-decisions--trade-offs)

---

## 1. System Architecture

The multi-teacher system follows a **single-controller orchestration pattern** within a Ray-based distributed training framework.

```
main_ppo.py (Hydra entry point)
    │
    ▼
TaskRunner (Ray remote actor)
    │
    ▼
RayPPOTrainer (orchestrator)
    │
    ├── actor_rollout_wg  (ActorRolloutRefWorker)
    │     ├── actor_module      (student policy, trainable)
    │     ├── rollout_module    (vLLM generation engine)
    │     ├── ref_module        (teacher #1 "math", frozen)
    │     ├── base_policy       (student base model, frozen)
    │     └── base_ref_policy   (teacher #2 "code", frozen)
    │
    ├── ref_policy_wg  (optional separate RefPolicy process)
    ├── critic_wg      (critic for GAE; unused in GRPO mode)
    ├── rm_wg          (reward model, optional)
    └── reward_fn      (rule-based reward manager)
```

All models are co-located within the `ActorRolloutRefWorker` class (`verl/verl/workers/fsdp_workers.py`) via `create_colocated_worker_cls` (`ray_trainer.py:769`). This colocation-first design maximizes GPU utilization through sequential execution with CPU offloading.

---

## 2. Component Interaction Map

### Training Loop Sequence (per step)

```
RayPPOTrainer.fit()                    [ray_trainer.py:1040-1300]
    │
    ├─1─► Rollout Worker (vLLM)         generate responses + rollout_log_probs
    │
    ├─2─► Reward Manager                rule-based scoring (teacher-agnostic)
    │
    ├─3─► Actor Worker                  recompute old_log_probs (FSDP FP32)
    │
    ├─4─► Ref Worker                    compute ref_log_prob (math teacher)
    │       sequential ↓
    ├─5─► Ref Worker                    compute base_ref_log_prob (code teacher)
    │       sequential ↓
    ├─6─► Actor Worker                  compute base_log_prob (student base)
    │
    ├─7─► Rollout Correction            IS weights from rollout vs old log-probs
    │
    └─8─► Actor Worker.update_policy()  per-sample teacher routing → reverse KL
                                         advantages → PPO loss → gradient update
```

Steps 4-6 are executed **strictly sequentially** — there is no parallelism between teacher forward passes. This is the primary throughput bottleneck for multi-teacher mode.

### Worker Communication

Workers communicate via `DataProto` objects passed through Ray. The `batch.union()` pattern progressively accumulates tensors from each computation phase:

- `batch.batch` — tensor data (log-probs, rewards, etc.)
- `batch.non_tensor_batch` — metadata like `opd_teacher` labels (numpy arrays)

---

## 3. FSDP Worker Model Colocation

In multi-teacher mode, `ActorRolloutRefWorker.__init__` (`fsdp_workers.py:840-900`) initializes five models:

| Model | Variable | Trainable | Memory Strategy |
|-------|----------|-----------|-----------------|
| Student (actor) | `self.actor_module_fsdp` | Yes | FSDP sharded |
| Student (rollout) | vLLM engine | No | vLLM managed (weights synced from actor) |
| Math teacher | `self.ref_module_fsdp` | No | FSDP + CPU param offload |
| Student base | `self.base_policy` | No | FSDP + CPU param offload |
| Code teacher | `self.base_ref_policy` | No | FSDP + CPU param offload |

All frozen models use `role="ref"` FSDP configuration with `param_offload=True` (line 858), enabling CPU offloading to reduce GPU memory pressure. During forward passes, parameters are loaded to GPU on-demand and resharded (`reshard(True)`) afterward.

### Initialization Code Pattern

```python
# fsdp_workers.py:844-869
base_model_path = self.config.model.get("base_model_path", None)
if base_model_path is not None and self._is_actor:
    self.base_module_fsdp = self._build_model_optimizer(
        model_path=local_base_path,
        fsdp_config=...,
        optim_config=None,  # No optimizer — frozen
        role="ref",          # CPU offload enabled
    )[0]
```

---

## 4. Rollout Correction Subsystem

The rollout correction module (`verl/verl/trainer/ppo/rollout_corr_helper.py`) is orthogonal to multi-teacher distillation. It addresses off-policy issues from:

1. **Precision mismatch**: vLLM uses BFloat16 for rollout; FSDP uses FP32 for training
2. **Model staleness**: Training on trajectories from older checkpoints

### Architecture

In **decoupled mode** (used by G-OPD):
- Three policies: `π_rollout` (vLLM BF16), `π_old` (FSDP FP32 recomputed), `π_θ` (current)
- IS weights from `log(π_old / π_rollout)` using `old_log_probs` and `rollout_log_probs`
- Applied centrally in the trainer before advantage computation (`ray_trainer.py:1263-1273`)

### Key Functions

| Function | Purpose |
|----------|---------|
| `compute_rollout_correction_and_add_to_batch()` | Batch-level entry point |
| `compute_rollout_correction_weights()` | Truncated IS weights (token or sequence level) |
| `compute_rollout_rejection_mask()` | Outlier filtering (token, sequence, or geometric) |
| `compute_offpolicy_metrics()` | Diagnostic metrics (KL, PPL, chi-squared, ESS) |

### Multi-Teacher Config

```yaml
rollout_is: token            # Token-level importance sampling
rollout_is_threshold: 5.0    # Fairly permissive truncation
bypass_mode: false           # Use decoupled 3-policy mode
```

---

## 5. Configuration Architecture

### Hydra Composition

Rooted at `verl/verl/trainer/config/ppo_trainer.yaml`:

```yaml
defaults:
  - actor@actor_rollout_ref.actor: dp_actor
  - ref@actor_rollout_ref.ref: dp_ref
  - rollout@actor_rollout_ref.rollout: rollout
  - model@actor_rollout_ref.model: hf_model
  - algorithm@algorithm.rollout_correction: rollout_correction
```

### Multi-Teacher Config Surface

Spread across three namespaces:

1. **Policy loss** (`actor.yaml:68-79`): `only_reverse_kl_advantages`, `lambda_vals`, `multi_teacher_distill`
2. **Model paths** (CLI overrides with `+` prefix): `model.base_model_path`, `ref.model.path`, `ref.model.base_model_path`
3. **Rollout correction** (`rollout_correction.yaml`): `rollout_is`, `rollout_is_threshold`, `bypass_mode`

### Naming Overload Issue

The config field `ref.model.base_model_path` serves a dual purpose:
- **Single-teacher G-OPD**: the reference model's base checkpoint (for reward correction normalization)
- **Multi-teacher ExOPD**: the **second teacher model** (code teacher)

These are semantically unrelated concepts sharing the same config field, creating confusion. From the shell script:

```bash
+actor_rollout_ref.ref.model.path=Qwen3-4B-Non-Thinking-RL-Math        # Teacher #1
+actor_rollout_ref.ref.model.base_model_path=Qwen3-4B-Non-Thinking-RL-Code  # Teacher #2 (NOT a "base")
```

---

## 6. Scalability Analysis

### Linear Scaling Cost

The design scales **linearly** in both time and memory with teacher count:

- **Time**: Each teacher adds one sequential forward pass. For N teachers: N teacher log-prob computations + 1 student base log-prob computation, all sequential.
- **Memory**: Each teacher is a separate FSDP-sharded module with CPU offload. GPU memory is time-multiplexed; CPU memory scales linearly. The batch grows by one `(batch_size, seq_length)` tensor per teacher.

### Hardcoded Two-Teacher Assumption

The routing logic (`dp_actor.py:505-514`) is hardcoded for exactly two teacher types:

```python
if teacher_type == "math":
    reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
elif teacher_type == "code":
    reverse_kl[i] = old_log_prob[i] - model_inputs["base_ref_log_prob"][i]
```

Adding a third teacher requires changes in four locations:
1. New config path and FSDP model initialization in `fsdp_workers.py`
2. New `compute_*_log_prob` method on the worker
3. New sequential call in `ray_trainer.py`
4. New branch in the routing logic in `dp_actor.py`

There is no generalized list-of-teachers abstraction.

---

## 7. Key Design Decisions & Trade-offs

### Decision 1: Reverse KL as Advantage Replacement

**Choice**: Replace reward-based advantages entirely with negative reverse KL divergence.

```python
advantages = -(reverse_kl)  # dp_actor.py:533
```

**Implication**: GRPO-computed advantages (`core_algos.py:264-328`) are calculated but then **overwritten** when `only_reverse_kl_advantages=True`. This wastes the GRPO computation cycle.

### Decision 2: Sequential Teacher Inference

**Choice**: Compute teacher log-probs sequentially, not in parallel.

**Rationale**: FSDP colocation means all models share the same GPU memory pool. Parallel forward passes would require simultaneous GPU memory for all unsharded parameters — infeasible for large models. Sequential execution with CPU offload allows time-multiplexing.

**Cost**: Wall-clock time scales linearly with teacher count.

### Decision 3: Data-Driven Teacher Routing

**Choice**: Teacher selection determined by the `opd_teacher` label in training data, not by runtime model selection.

**Pros**:
- Simple, deterministic — no routing overhead at runtime
- Data mixing ratio directly controls teacher influence ratio

**Cons**:
- No dynamic routing based on model confidence or difficulty
- Requires pre-labeled datasets

### Decision 4: Base Model Normalization

**Choice**: When `lambda_vals ≠ 1.0`, reverse KL is computed as a corrected quantity:

```
reverse_kl = (log π_student - log π_base) - λ × (log π_teacher - log π_base)
```

This isolates "learned improvement" over the base for both student and teacher. The `λ` parameter then scales the teacher's improvement (extrapolation when `λ > 1`).

**Cost**: Requires maintaining an additional frozen copy of the student's base model.

### Decision 5: Hybrid Engine Colocation

**Choice**: Actor, rollout, reference, and all base models share a single `ActorRolloutRefWorker`.

**Benefit**: Maximizes GPU utilization, eliminates inter-process tensor transfer.

**Risk**: Worker class manages 5 models, making it very complex. Memory management requires careful FSDP resharding orchestration.
