#!/bin/bash
# check-expert-update.sh
# Hook: Runs after Write/Edit to remind developer to update relevant expert agents

CHANGED_FILE="$1"

# Map changed files to relevant expert agents
case "$CHANGED_FILE" in
    */trainer/ppo/core_algos.py|*/trainer/ppo/ray_trainer.py|*/trainer/ppo/reward.py)
        echo "📝 Consider updating .claude/agents/algorithm-expert.md if algorithm behavior changed"
        ;;
    */workers/fsdp_workers.py|*/utils/fsdp_utils.py|*/workers/engine/fsdp/*)
        echo "📝 Consider updating .claude/agents/fsdp-engine-expert.md if FSDP behavior changed"
        ;;
    */workers/actor/dp_actor.py|*/workers/critic/dp_critic.py)
        echo "📝 Consider updating .claude/agents/algorithm-expert.md if policy/value update logic changed"
        ;;
    */trainer/ppo/rollout_corr_helper.py|*/trainer/ppo/ref_input_utils.py)
        echo "📝 Consider updating .claude/agents/algorithm-expert.md if G-OPD/ExOPD logic changed"
        ;;
    */workers/rollout/*)
        echo "📝 Consider updating .claude/agents/rollout-engine-expert.md if rollout behavior changed"
        ;;
    */single_controller/*|*/trainer/main_ppo.py)
        echo "📝 Consider updating .claude/agents/ray-trainer-expert.md if orchestration changed"
        ;;
    */workers/reward_manager/*|*/utils/reward_score/*)
        echo "📝 Consider updating .claude/agents/reward-data-expert.md if reward system changed"
        ;;
    */workers/sharding_manager/*)
        echo "📝 Consider updating .claude/agents/fsdp-engine-expert.md if weight sync changed"
        ;;
esac
