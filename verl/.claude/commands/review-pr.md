# Review Pull Request

Dynamic PR review with agent allocation based on change types.

## Arguments
- No args: Review current branch against main
- `<PR_NUMBER>`: Review specific PR
- `--quick`: Use Sonnet for all reviews (faster)

## Phases

### Phase 1: PR Analysis
```bash
# Get PR details
gh pr view <number> --json title,body,files,additions,deletions

# Get diff
gh pr diff <number>
```

**Detect change types** using the patterns in `data/review-pr-change-types.md`.

### Phase 2: Dynamic Agent Planning

Allocate review agents based on risk level:

| Risk Level | Model | Trigger Files |
|-----------|-------|---------------|
| CRITICAL | Opus | `core_algos.py`, `fsdp_workers.py`, `ray_trainer.py`, `protocol.py` |
| HIGH | Opus | `dp_actor.py`, `dp_critic.py`, `rollout_corr_helper.py`, `ref_input_utils.py` |
| MEDIUM | Sonnet | Worker configs, reward functions, utility modules, rollout engines |
| LOW | Haiku | Tests, docs, YAML-only changes |

### Phase 3: Execute Review Tasks

For each detected change type:
1. Spawn a review agent with the appropriate model
2. Agent reviews the specific files in its scope
3. Each agent produces findings

### Phase 4: Confidence Scoring & Summary

Aggregate findings into:
```markdown
## PR Review: <title>

### Risk Assessment
Overall: 🟢/🟡/🔴

### Findings by Area
#### <Area> (Risk: <level>)
- 🔴 Critical: ...
- 🟡 Warning: ...
- 🔵 Suggestion: ...

### Recommendation
APPROVE / REQUEST_CHANGES / COMMENT
```
