# verl Claude Configuration Documentation

Welcome to the verl `.claude` configuration documentation. This directory contains guides for using the specialized agents and skills.

## Documentation Files

### Agent Guides
- **[agents-guide.md](agents-guide.md)** - Detailed guide for agents 1-5 (Algorithm, FSDP, Ray Trainer, Rollout, Reward & Data)
- **[agents-guide-part2.md](agents-guide-part2.md)** - Detailed guide for agents 6-8 (Planner, Code Verifier, Simple Code Reviewer)

### Skills Guides
- **[skills-guide.md](skills-guide.md)** - Detailed guide for skills 1-3 (Add Advantage Estimator, Add Policy Loss, Add Dataset)
- **[skills-guide-part2.md](skills-guide-part2.md)** - Detailed guide for skills 4-6 (Add Reward, Add Unit Tests, Debug Distributed)

### Quick Reference
- **[quick-reference.md](quick-reference.md)** - Quick lookup table for all agents and skills

## What's in the .claude Configuration?

The verl `.claude` configuration provides:

### 🤖 8 Specialized Agents
Expert assistants that automatically activate based on your questions:
1. **Algorithm Expert** (Opus) - RL algorithms, G-OPD/ExOPD, advantage estimators, policy losses
2. **FSDP Engine Expert** (Opus) - FSDP configuration, weight sync, checkpointing
3. **Ray Trainer Expert** (Opus) - Training orchestration, worker groups, dispatch patterns
4. **Rollout Engine Expert** (Sonnet) - vLLM/SGLang/HF rollout, sampling configuration
5. **Reward & Data Expert** (Sonnet) - Reward functions, datasets, reward managers
6. **Planner** (Opus) - Implementation planning for complex changes
7. **Code Verifier** (Haiku) - Automated code quality checking
8. **Simple Code Reviewer** (Haiku) - Quick code reviews

### 📚 6 Skills
Step-by-step guides for common tasks:
1. **add-advantage-estimator** - Add new advantage estimators
2. **add-policy-loss** - Add new policy loss functions
3. **add-dataset** - Add new dataset loaders
4. **add-reward** - Add new reward functions
5. **add-unit-tests** - Add unit tests following conventions
6. **debug-distributed** - Debug distributed training issues

### 📋 4 Rules
Coding standards automatically applied to relevant files:
- **code-style.md** - verl code conventions
- **testing.md** - Test structure and patterns
- **distributed.md** - Distributed training standards
- **api-config.md** - Configuration conventions

### 🔧 3 Commands
Workflow automation:
- **gen-commit-msg** - Generate conventional commit messages
- **create-pr** - Full PR creation workflow
- **review-pr** - Dynamic PR review with risk-based agent allocation

## Quick Start

### Ask Questions Naturally
```
"How does G-OPD's reward scaling work?"
"How do I add a new advantage estimator?"
"My training is hanging, how do I debug it?"
```

The system will automatically activate the appropriate agent.

### Invoke Skills Explicitly
```
/add-reward my_custom_reward
/add-dataset my_math_dataset
/debug-distributed
```

### Get Implementation Plans
For complex changes (3+ files), the Planner agent automatically activates:
```
"I want to add support for custom reward shaping"
"Help me implement a new training loop variant"
```

## How It Works

```
Your Question
    ↓
System analyzes topic
    ↓
Activates relevant agent(s)
    ↓
Agent reads source files
    ↓
Provides expert guidance with:
    - File references
    - Code examples
    - Configuration examples
    - Best practices
    ↓
You implement changes
    ↓
Code Verifier checks quality
```

## Configuration Structure

```
.claude/
├── agents/          # 8 specialized expert agents
├── commands/        # 3 workflow automation commands
├── data/            # PR review risk classification
├── docs/            # This documentation (you are here)
├── hooks/           # Automation scripts
├── rules/           # 4 coding standards
├── skills/          # 6 step-by-step guides
└── settings.json    # Hook configuration
```

## Getting Help

- **For agent details**: See [agents-guide.md](agents-guide.md) and [agents-guide-part2.md](agents-guide-part2.md)
- **For skill details**: See [skills-guide.md](skills-guide.md) and [skills-guide-part2.md](skills-guide-part2.md)
- **For quick lookup**: See [quick-reference.md](quick-reference.md)

## Examples

### Example 1: Adding a New Algorithm
```
User: "I want to add a new advantage estimator called 'my_custom'"

System activates: Planner + Algorithm Expert

Planner: Analyzes scope, identifies files to change
Algorithm Expert: Provides implementation details
User: Follows /add-advantage-estimator skill
Code Verifier: Checks implementation quality
```

### Example 2: Debugging Training Issues
```
User: "Training hangs after 10 steps"

System activates: Ray Trainer Expert + debug-distributed skill

Provides:
- Diagnostic steps
- Environment variables to set
- Common causes and solutions
- Debugging tools (Ray dashboard, NCCL tests)
```

### Example 3: Understanding G-OPD
```
User: "How does multi-teacher distillation work in ExOPD?"

System activates: Algorithm Expert

Explains:
- ExOPD routing via opd_teacher field
- Math teacher vs code teacher selection
- Configuration parameters
- References to dp_actor.py implementation
```

## Contributing

When you modify verl code, the hook system automatically reminds you to update relevant expert agents if their domain changed.
