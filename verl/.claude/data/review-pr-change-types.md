# PR Review Change Type Detection

## Risk Levels & Model Selection

### CRITICAL (Use Opus)
| Pattern | Description |
|---------|-------------|
| `verl/trainer/ppo/core_algos.py` | Core advantage estimators and policy losses |
| `verl/trainer/ppo/ray_trainer.py` | Main training orchestration loop |
| `verl/protocol.py` | DataProto - central data exchange object |
| `verl/workers/fsdp_workers.py` | Composite FSDP worker |
| `verl/workers/megatron_workers.py` | Megatron backend workers |
| `verl/single_controller/` | Ray RPC dispatch infrastructure |
| `verl/workers/actor/dp_actor.py` | PPO actor with G-OPD logic |
| `verl/trainer/ppo/rollout_corr_helper.py` | G-OPD importance sampling correction (core research code) |

### HIGH (Recommend Opus)
| Pattern | Description |
|---------|-------------|
| `verl/workers/actor/` | Actor implementations |
| `verl/workers/critic/` | Critic implementations |
| `verl/workers/sharding_manager/` | Weight sync FSDP ↔ vLLM/SGLang |
| `verl/trainer/ppo/ref_input_utils.py` | Cross-model ref tokenization |
| `verl/workers/engine/` | Training engine backends |
| `verl/utils/fsdp_utils.py` | FSDP wrapping utilities |
| `verl/utils/checkpoint/` | Checkpoint management |

### MEDIUM (Use Sonnet)
| Pattern | Description |
|---------|-------------|
| `verl/workers/rollout/` | Rollout engine implementations |
| `verl/workers/reward_manager/` | Reward computation |
| `verl/workers/reward_model/` | Model-based reward scoring |
| `verl/workers/config/` | Worker configuration dataclasses |
| `verl/trainer/config/` | Hydra YAML configurations |
| `verl/utils/reward_score/` | Rule-based reward functions |
| `verl/utils/dataset/` | Dataset loading utilities |
| `verl/utils/torch_functional.py` | Tensor utility functions |
| `verl/utils/seqlen_balancing.py` | Workload distribution |
| `verl/utils/tracking.py` | Logging/tracking utilities |
| `verl/models/` | Model implementations and patches |
| `verl/base_config.py` | Base configuration |

### LOW (Use Haiku)
| Pattern | Description |
|---------|-------------|
| `tests/` | Test files |
| `*.md` | Documentation |
| `*.yaml` only | Config-only changes |
| `verl/version/` | Version changes |
| `scripts/` | Utility scripts |
| `examples/` | Example configurations |
