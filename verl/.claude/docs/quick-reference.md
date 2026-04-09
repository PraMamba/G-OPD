# Quick Reference Guide

Fast lookup for all agents and skills in the verl `.claude` configuration.

## Agents Quick Reference

| Agent | Model | Activation | Key Topics |
|-------|-------|------------|------------|
| **Algorithm Expert** | Opus | Auto | PPO, GRPO, G-OPD, ExOPD, advantage estimators, policy losses, reward scaling, KL penalty, rollout correction |
| **FSDP Engine Expert** | Opus | Auto | FSDP config, worker init, weight sync (ShardingManager), checkpointing, gradient accumulation |
| **Ray Trainer Expert** | Opus | Auto | Training loop, worker groups, resource pools, dispatch patterns, Ray orchestration |
| **Rollout Engine Expert** | Sonnet | Auto | vLLM, SGLang, HF rollout, sampling params, weight transfer, async server |
| **Reward & Data Expert** | Sonnet | Auto | Reward functions, reward managers, datasets, RLHFDataset, rule-based vs model-based rewards |
| **Planner** | Opus | Proactive | Implementation planning, multi-file changes, architecture decisions, scope analysis |
| **Code Verifier** | Haiku | Proactive | Automated quality checks, ruff lint/format, test execution, verification reports |
| **Simple Code Reviewer** | Haiku | Manual | Quick code reviews, style checks, pattern validation, safety checks |

## Skills Quick Reference

| Skill | Invocation | Purpose | Key Steps |
|-------|------------|---------|-----------|
| **add-advantage-estimator** | `/add-advantage-estimator <name>` | Add new advantage estimator | 1. Implement in core_algos.py with @register_adv_est<br>2. Add to config<br>3. Test registration<br>4. Add unit tests |
| **add-policy-loss** | `/add-policy-loss <name>` | Add new policy loss | 1. Implement in core_algos.py with @register_policy_loss<br>2. Configure in dp_actor.py<br>3. Add unit tests |
| **add-dataset** | `/add-dataset <name>` | Add new dataset loader | 1. Create dataset file<br>2. Register via Hydra config<br>3. Ensure required fields<br>4. Add tests |
| **add-reward** | `/add-reward <name>` | Add new reward function | 1. Create reward file with compute_score()<br>2. Register in reward manager<br>3. Handle blocking ops<br>4. Add tests |
| **add-unit-tests** | `/add-unit-tests` | Add tests following conventions | 1. Choose pytest or unittest<br>2. Follow naming conventions<br>3. Handle GPU tests<br>4. Cover edge cases |
| **debug-distributed** | `/debug-distributed` | Debug distributed training | 1. Identify issue (hang/OOM/wrong results/comm errors)<br>2. Enable debug logging<br>3. Use diagnostic tools<br>4. Apply fixes |

## Common Questions → Agent Mapping

| Question Type | Activates Agent |
|---------------|-----------------|
| "How does [algorithm] work?" | Algorithm Expert |
| "How do I configure FSDP?" | FSDP Engine Expert |
| "Training loop flow?" | Ray Trainer Expert |
| "vLLM configuration?" | Rollout Engine Expert |
| "Add custom reward?" | Reward & Data Expert |
| "Plan implementation for X" | Planner |
| "Check my code quality" | Code Verifier |
| "Quick review of this change" | Simple Code Reviewer |

## Key Files Reference

### Core Algorithm Files
- `verl/trainer/ppo/core_algos.py` - Advantage estimators, policy losses
- `verl/workers/actor/dp_actor.py` - Actor with G-OPD logic
- `verl/trainer/ppo/ray_trainer.py` - Main training loop
- `verl/trainer/ppo/rollout_corr_helper.py` - Importance sampling
- `verl/trainer/ppo/ref_input_utils.py` - Cross-model tokenization

### Worker Files
- `verl/workers/fsdp_workers.py` - Composite FSDP worker
- `verl/workers/actor/` - Actor implementations
- `verl/workers/critic/` - Critic implementations
- `verl/workers/rollout/` - Rollout engines
- `verl/workers/sharding_manager/` - Weight sync

### Configuration Files
- `verl/trainer/config/` - Hydra YAML configs
- `verl/workers/config/` - Worker dataclass configs
- `verl/base_config.py` - Base configuration

### Utilities
- `verl/protocol.py` - DataProto
- `verl/utils/fsdp_utils.py` - FSDP utilities
- `verl/utils/reward_score/` - Reward functions
- `verl/utils/dataset/` - Dataset loaders
- `verl/utils/profiler/` - Profiling tools

## Registry Patterns

### Advantage Estimators
```python
@register_adv_est(name="my_estimator")
def compute_my_estimator_advantage(...):
    ...
```

### Policy Losses
```python
@register_policy_loss(name="my_loss")
def compute_my_loss_policy_loss(...):
    ...
```

### Reward Managers
```python
@register("my_manager")
class MyRewardManager:
    ...
```

## Common CLI Overrides

```bash
# Change advantage estimator
algorithm.adv_estimator=grpo

# Change policy loss
actor_rollout_ref.actor.policy_loss.name=vanilla

# G-OPD configuration
+actor_rollout_ref.model.base_model_path=Qwen/Qwen3-1.7B
actor_rollout_ref.actor.policy_loss.lambda_vals=1.25

# Multi-teacher ExOPD
actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true
```

## Debug Commands

```bash
# Enable distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export RAY_DEDUP_LOGS=0

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check NCCL
python -c "import torch; print(torch.cuda.nccl.version())"

# Ray dashboard
ray start --head --dashboard-host=0.0.0.0
# Access at http://localhost:8265
```

## Test Commands

```bash
# Run all tests
cd verl && pytest tests/

# Run specific test file
pytest tests/trainer/ppo/test_core_algos.py

# Run with GPU
pytest tests/special_e2e/

# Lint and format
ruff check . --fix
ruff format .
```
