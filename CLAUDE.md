# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

G-OPD (Generalized On-Policy Distillation) is a research framework for training LLMs using reinforcement learning with knowledge distillation. It introduces a reward scaling factor and flexible reference model. ExOPD (On-Policy Distillation with Reward Extrapolation) is the enhanced multi-teacher variant. Built on verl v0.6.1.

## Repository Layout

- `verl/` — Main training framework (installable package under `verl/verl/`)
- `math_eval/` — Math reasoning evaluation (vLLM-based)
- `code_eval/` — Code generation evaluation (EvalPlus, LiveCodeBench)
- `data/` — Evaluation datasets (AIME24, AIME25, HMMT25)

## Installation

```bash
conda create -n verl python==3.10
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

## Key Commands

### Training
```bash
# G-OPD single-teacher distillation
cd verl/examples/g_opd/ && bash run_qwen3-4b-g-opd.sh

# ExOPD multi-teacher distillation
bash run_qwen3-4b-g-opd-multi-teacher.sh

# GRPO baseline
cd verl/examples/grpo_trainer/ && bash run_qwen3-4b-math.sh
```

### Evaluation
```bash
# Math eval
cd math_eval/ && sh run_eval_math.sh

# Code eval (EvalPlus)
bash code_eval/scripts/run_evalplus.sh humaneval <MODEL_PATH> 0 1.0 1.0 4

# Code eval (LiveCodeBench)
bash code_eval/scripts/run_lcb_gen.sh --model Qwen3-4B --local_model_path <MODEL_PATH>
```

### Tests & Linting
```bash
cd verl/ && pytest tests/                # Run tests
cd verl/ && ruff check .                 # Lint (line-length=120, rules: E,F,UP,B,I,G)
```

## Architecture

The training system is Ray-based with these distributed worker roles:

- **Main Controller** (`verl/trainer/main_ppo.py`) — Entry point, Hydra config driven
- **Ray Trainer** (`verl/trainer/ppo/ray_trainer.py`) — Orchestrates distributed workers
- **Actor Worker** (`verl/workers/actor/`) — Student model policy learning
- **Critic Worker** (`verl/workers/critic/`) — Value function estimation
- **Rollout Worker** (`verl/workers/rollout/`) — Generation via vLLM or SGLang
- **Reference Worker** — Teacher model for distillation KL computation
- **Reward Manager** (`verl/workers/reward_manager/`) — Reward computation from teacher outputs

Core algorithms live in `verl/trainer/ppo/core_algos.py` (PPO/GRPO with distillation modifications).

## Configuration

Hydra YAML configs in `verl/trainer/config/`. Override via CLI:
```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
  trainer.n_gpus_per_node=8
```

Key G-OPD settings: `only_reverse_kl_advantages=True`, `lambda_vals` (reward scaling, typically 1.25), `multi_teacher_distill=true`, `rollout_correction` (importance sampling).

## Key Dependencies

Ray (>=2.41.0), transformers, vLLM (>=0.8.5,<=0.11.0), hydra-core, tensordict, peft, wandb, math-verify.
