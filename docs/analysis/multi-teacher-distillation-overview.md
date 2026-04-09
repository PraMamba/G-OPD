# 多教师蒸馏：实现概述

本文档全面分析了 G-OPD 如何实现多教师蒸馏（ExOPD），追踪从配置到训练的完整数据流。

## 目录

- [1. 概念总结](#1-概念总结)
- [2. 涉及的模型](#2-涉及的模型)
- [3. 配置](#3-配置)
- [4. 数据流](#4-数据流)
- [5. 数学公式](#5-数学公式)
- [6. 关键源文件](#6-关键源文件)

---

## 1. 概念总结

ExOPD 将单教师 G-OPD 扩展为多教师蒸馏框架。其核心思路是根据训练数据中每条样本嵌入的 `opd_teacher` 标签，将样本**路由到特定领域的教师模型**。目前支持两个教师：

- **数学教师** — 数学领域的 RL 微调模型（例如 `Qwen3-4B-Non-Thinking-RL-Math`）
- **代码教师** — 代码领域的 RL 微调模型（例如 `Qwen3-4B-Non-Thinking-RL-Code`）

学生策略同时从两个教师中蒸馏，优势函数（advantage）计算为向对应教师方向的负反向 KL 散度。**奖励缩放因子** (`lambda_vals`，通常为 1.25) 控制超越教师能力的外推强度。

---

## 2. 涉及的模型

多教师蒸馏需要加载 **五个** 模型实例：

| 模型 | 配置路径 | 输出键 | 用途 |
|------|----------|--------|------|
| 演员（学生） | `actor_rollout_ref.model.path` | `old_log_probs` | 可训练的学生策略 |
| 采样引擎（vLLM） | 与演员相同 | `rollout_log_probs` | BF16 生成引擎 |
| 参考（数学教师） | `actor_rollout_ref.ref.model.path` | `ref_log_prob` | 数学领域教师 |
| 演员基础模型 | `actor_rollout_ref.model.base_model_path` | `base_log_prob` | 学生的预训练基础模型（归一化） |
| 参考基础模型（代码教师） | `actor_rollout_ref.ref.model.base_model_path` | `base_ref_log_prob` | 代码领域教师 |

所有冻结模型（参考、演员基础、参考基础）均使用 FSDP 的 CPU 参数卸载（`role="ref"`）来管理 GPU 显存。

---

## 3. 配置

### Shell 脚本（`verl/examples/g_opd/run_qwen3-4b-g-opd-multi-teacher.sh`）

```bash
# 模型路径
actor_rollout_ref.model.path=Qwen/Qwen3-4B                              # 学生
+actor_rollout_ref.model.base_model_path=Qwen/Qwen3-4B                  # 学生基础模型
+actor_rollout_ref.ref.model.path=Qwen3-4B-Non-Thinking-RL-Math         # 数学教师
+actor_rollout_ref.ref.model.base_model_path=Qwen3-4B-Non-Thinking-RL-Code  # 代码教师

# 多教师标志
actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True
actor_rollout_ref.actor.policy_loss.lambda_vals=1.25
actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true

# 采样校正（重要性采样）
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=5.0
algorithm.rollout_correction.bypass_mode=false
```

### 数据类配置（`verl/verl/workers/config/actor.py:44-57`）

```python
@dataclass
class PolicyLossConfig(BaseConfig):
    only_reverse_kl_advantages: bool = False  # 使用反向 KL 作为优势函数
    lambda_vals: float = 1.0                   # 奖励缩放（1.0=OPD，>1.0=ExOPD）
    multi_teacher_distill: bool = False        # 启用多教师路由
```

注意：`base_model_path` 条目使用 `+` 前缀，这是 Hydra 的临时覆盖 — 这些字段未在基础 YAML schema 中声明。

---

## 4. 数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 1：数据加载                                                    │
│   数据集 parquet -> extra_info.opd_teacher -> non_tensor_batch      │
│   (rl_dataset.py:454-455)                                           │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 2：采样生成                                                    │
│   学生模型通过 vLLM 生成回答 -> rollout_log_probs                    │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 3：奖励计算                                                    │
│   基于规则的评分（数学验证 / 代码执行）                               │
│   与教师无关 (reward.py:170-188)                                    │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 4：旧对数概率重计算                                             │
│   演员在 FSDP FP32 下前向传播 -> old_log_probs                       │
│   (ray_trainer.py:1147-1159)                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 5：教师对数概率计算（顺序执行）                                  │
│   步骤 A：数学教师  -> ref_log_prob                                  │
│   步骤 B：代码教师  -> base_ref_log_prob                             │
│   步骤 C：学生基础  -> base_log_prob                                 │
│   (ray_trainer.py:1186-1233)                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 6：逐样本教师路由与优势函数计算                                  │
│   根据 opd_teacher 标签路由每个样本 -> 计算反向 KL                    │
│   advantages = -(reverse_kl)                                        │
│   (dp_actor.py:494-533)                                             │
├─────────────────────────────────────────────────────────────────────┤
│ 阶段 7：策略损失与梯度更新                                           │
│   使用反向 KL 优势的标准 PPO 裁剪损失                                │
│   可选的采样校正重要性采样权重                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 数学公式

### 标准 OPD（`lambda_vals = 1.0`）

对于分配给教师 `k` 的序列 `i` 中每个词元位置 `t`：

```
A(i,t) = log π_teacher_k(x_t | x_<t) - log π_student(x_t | x_<t)
```

学生被推向教师的分布。

### G-OPD / ExOPD（`lambda_vals ≠ 1.0`）

```
A(i,t) = -[(log π_student - log π_base) - λ × (log π_teacher_k - log π_base)]
```

其中：
- `(log π_teacher_k - log π_base)` 衡量教师相对于基础模型的提升幅度
- `λ > 1.0` 外推超越教师的提升（增强蒸馏效果）
- `λ < 1.0` 内插（减弱教师影响）
- 基础模型为学生和教师提供共同的锚定点

### 教师路由

对于样本 `i`，根据 `opd_teacher[i]` 标签：
- `"math"` → 使用 `ref_log_prob`（数学教师）
- `"code"` → 使用 `base_ref_log_prob`（代码教师）

路由是**数据驱动**的 — 教师分配在训练数据集中预先确定，而非运行时动态决定。

---

## 6. 关键源文件

| 文件 | 行号 | 作用 |
|------|------|------|
| `verl/verl/workers/actor/dp_actor.py` | 486-533 | 核心多教师反向 KL 计算 |
| `verl/verl/trainer/ppo/ray_trainer.py` | 1186-1233 | 顺序教师对数概率编排 |
| `verl/verl/workers/fsdp_workers.py` | 840-900 | 基础模型初始化（5个模型） |
| `verl/verl/utils/dataset/rl_dataset.py` | 454-455 | 从数据中提取 `opd_teacher` 字段 |
| `verl/verl/workers/config/actor.py` | 44-57 | PolicyLossConfig 数据类 |
| `verl/verl/trainer/ppo/rollout_corr_helper.py` | — | 重要性采样校正 |
| `verl/verl/trainer/ppo/ref_input_utils.py` | 36+ | 不同教师分词器的重新分词 |
| `verl/examples/g_opd/run_qwen3-4b-g-opd-multi-teacher.sh` | — | 多教师训练脚本 |

另见：
- [架构分析](./architecture-analysis.md)
- [代码审查](./code-review.md)
