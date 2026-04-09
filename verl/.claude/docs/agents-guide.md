# verl Claude Agents Guide

This guide explains how to use the specialized agents in the verl `.claude` configuration.

## Overview

The verl `.claude` configuration includes 8 specialized agents that provide expert guidance on different aspects of the codebase. Agents are automatically activated based on the topic of your question, or you can explicitly invoke them.

## Agent Activation

Agents activate automatically when you ask questions related to their domain. You can also explicitly request an agent by mentioning its area of expertise.

---

## 1. Algorithm Expert

**Model**: Opus
**File**: `.claude/agents/algorithm-expert.md`

### Purpose
Expert on verl's RL training algorithms, including PPO, GRPO, and G-OPD/ExOPD distillation methods.

### When It Activates
- Questions about RL algorithms (PPO, GRPO, RLOO, REINFORCE++, GAE, ReMax, OPO, GPG)
- Advantage estimation and policy loss computation
- G-OPD/ExOPD distillation mechanisms
- Reward scaling and KL penalty
- Rollout correction / importance sampling

### Example Questions
```
"How does G-OPD's reward scaling work?"
"What's the difference between GRPO and RLOO advantage estimators?"
"How do I configure multi-teacher distillation for ExOPD?"
"Explain the rollout correction mechanism"
"What policy losses are available in verl?"
```

### What It Knows
- All advantage estimators in `core_algos.py` (gae, grpo, rloo, etc.)
- All policy losses (vanilla, gspo, gpg, clip_cov, kl_cov, geo_mean, rollout_correction)
- G-OPD mechanism: `only_reverse_kl_advantages=True`, `lambda_vals` for reward extrapolation
- ExOPD multi-teacher routing via `opd_teacher` field
- KL penalty functions and loss aggregation modes
- Rollout correction for off-policy importance sampling

### Key Files It References
- `verl/trainer/ppo/core_algos.py`
- `verl/workers/actor/dp_actor.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/rollout_corr_helper.py`
- `verl/trainer/ppo/ref_input_utils.py`

---

## 2. FSDP Engine Expert

**Model**: Opus
**File**: `.claude/agents/fsdp-engine-expert.md`

### Purpose
Expert on verl's FSDP-based distributed training engine, including worker initialization, weight synchronization, and checkpointing.

### When It Activates
- Questions about FSDP configuration
- Worker model initialization and wrapping
- Weight synchronization between FSDP and vLLM/SGLang
- Gradient accumulation and micro-batching
- Checkpoint management (save/load)
- ShardingManager patterns

### Example Questions
```
"How do I configure FSDP wrapping for my model?"
"How does weight sync work between FSDP and vLLM?"
"How do I save and load FSDP checkpoints?"
"What's the ShardingManager and when is it used?"
"How do I configure gradient accumulation?"
```

### What It Knows
- `ActorRolloutRefWorker` composite worker pattern
- Registered methods: `generate_sequences()`, `compute_log_prob()`, `update_actor()`
- Weight synchronization via `ShardingManager`:
  - `sync_weights_to_rollout()` before generation
  - `sync_weights_from_rollout()` after generation
- FSDP utilities in `verl/utils/fsdp_utils.py`
- Checkpoint management with `FSDPCheckpointManager`

### Key Files It References
- `verl/workers/fsdp_workers.py`
- `verl/workers/actor/dp_actor.py`
- `verl/workers/critic/dp_critic.py`
- `verl/workers/sharding_manager/`
- `verl/utils/fsdp_utils.py`
- `verl/utils/checkpoint/`

---

## 3. Ray Trainer Expert

**Model**: Opus
**File**: `.claude/agents/ray-trainer-expert.md`

### Purpose
Expert on verl's Ray-based distributed training orchestration layer, including worker groups, resource pools, and the training loop.

### When It Activates
- Questions about RayPPOTrainer and training loop flow
- Worker group management and resource pools
- Single controller dispatch patterns
- Worker colocation strategies
- Data flow between trainer and workers
- Training configuration (Hydra)

### Example Questions
```
"How does the Ray-based training loop work?"
"What are worker roles and how are they colocated?"
"How do I configure resource pools?"
"What's the data flow in a training step?"
"How do dispatch decorators work?"
```

### What It Knows
- Training loop flow: gen → reward → ref → critic → advantage → update
- Worker roles: Actor, Rollout, ActorRollout, Critic, RefPolicy, RewardModel, ActorRolloutRef
- Dispatch modes: `DP_COMPUTE_PROTO`, `DP_COMPUTE_METRIC`, `RANK_ZERO`, `ONE_TO_ALL`
- Resource pool management and worker colocation
- Entry point: `main_ppo.py` with Hydra configuration
- Metric aggregation pipeline

### Key Files It References
- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/utils.py`
- `verl/single_controller/base/`
- `verl/single_controller/ray/`
- `verl/workers/fsdp_workers.py`

---

## 4. Rollout Engine Expert

**Model**: Sonnet
**File**: `.claude/agents/rollout-engine-expert.md`

### Purpose
Expert on verl's rollout/generation engines (vLLM, SGLang, HF), including configuration and weight synchronization.

### When It Activates
- Questions about text generation / rollout configuration
- vLLM integration and SPMD mode
- SGLang rollout backend
- Rollout sampling parameters (temperature, top_p, top_k)
- Weight synchronization for inference
- Async server mode for agent loops

### Example Questions
```
"How do I configure vLLM for rollout?"
"What's the difference between vLLM and SGLang backends?"
"How do I adjust sampling parameters?"
"How does weight transfer work for inference?"
"How do I use async server mode?"
```

### What It Knows
- Supported backends: vLLM (SPMD), SGLang, HF, Async server
- vLLM SPMD mode with tensor parallelism
- Weight transfer via `ShardingManager`
- Sampling configuration: temperature, top_p, top_k, max_new_tokens
- Dynamic batching with `prepare_dynamic_batch()`

### Key Files It References
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/workers/rollout/sglang_rollout/`
- `verl/workers/rollout/hf_rollout.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/sharding_manager/`

---

## 5. Reward & Data Expert

**Model**: Sonnet
**File**: `.claude/agents/reward-data-expert.md`

### Purpose
Expert on verl's reward system and dataset management, including reward functions and data loading.

### When It Activates
- Questions about reward function implementation
- Reward managers (Naive, Prime, DAPO)
- Custom reward functions
- Dataset loading and processing
- Rule-based vs model-based rewards

### Example Questions
```
"How do I add a custom reward function?"
"What reward managers are available?"
"How do I load a new dataset?"
"What's the difference between rule-based and model-based rewards?"
"How do I configure the reward pipeline?"
```

### What It Knows
- Reward manager registry: NaiveRewardManager, PrimeRewardManager, DAPORewardManager
- Rule-based reward functions in `verl/utils/reward_score/`
- Dataset system with `RLHFDataset`
- Reward function signature: `compute_score(data_source, solution_str, ground_truth, extra_info)`
- Custom reward function configuration via YAML

### Key Files It References
- `verl/workers/reward_manager/naive.py`
- `verl/workers/reward_manager/prime.py`
- `verl/utils/reward_score/`
- `verl/utils/dataset/`
- `verl/trainer/ppo/reward.py`
