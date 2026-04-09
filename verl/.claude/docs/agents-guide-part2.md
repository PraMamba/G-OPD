---

## 6. Planner

**Model**: Opus
**File**: `.claude/agents/planner.md`
**Activation**: PROACTIVE

### Purpose
Expert planner for the verl codebase. Researches, analyzes dependencies, and produces implementation plans. **Read-only** — never modifies code directly.

### When It Activates
**Automatically activates** when:
- Multi-file changes (3+ files) are needed
- New features are being designed
- Architectural decisions need to be made
- The scope of a change is unclear

### Example Scenarios
```
"I want to add a new advantage estimator"
"Help me implement custom reward shaping"
"I need to modify the training loop to support X"
"How should I refactor the worker initialization?"
```

### What It Does

**Phase 1: Understanding**
- Clarifies requirements with specific questions (max 2-3)
- Identifies scope: which layers affected (config, worker, algorithm, utility)
- Finds existing patterns to follow

**Phase 2: Research**
- Finds similar implementations in the codebase
- Identifies callers and dependencies
- Checks existing tests
- Checks configuration requirements

**Phase 3: Plan Output**
- **Quick Path** (2-3 files): Summary + Changes table + Steps
- **Full Plan** (complex): Summary + Architecture Impact + Changes + Steps + Patterns + Risks + Testing

### verl-Specific Considerations
- All worker communication goes through `DataProto`
- New algorithms register via `@register_adv_est()` or `@register_policy_loss()`
- Config changes need both YAML and dataclass updates
- Worker changes need dispatch decorator registration
- New reward functions use the reward manager registry

---

## 7. Code Verifier

**Model**: Haiku
**File**: `.claude/agents/code-verifier.md`
**Activation**: PROACTIVE

### Purpose
Fast, automated code quality checker for the verl codebase.

### When It Activates
**Automatically activates**:
- After code changes (Write/Edit)
- Before commits
- After implementing features

### What It Does

**Step 1: Identify Changed Files**
Categorizes changes: Python, Config (YAML), Markdown

**Step 2: Run Formatting & Linting**
```bash
cd /home/scbjtfy/G-OPD/verl
ruff check . --fix  # Auto-fix
ruff format .       # Format code
```

**Step 3: Run Relevant Tests**
- Checks GPU availability first
- Runs tests based on changed files:
  - `core_algos.py` → `pytest tests/trainer/ppo/`
  - `protocol.py` → `pytest tests/test_protocol_on_cpu.py`
  - `workers/actor/` → `pytest tests/workers/actor/`
  - General: `pytest tests/ -x --timeout=120`

**Step 4: Report Results**
Provides a verification report with status for each check (lint, format, tests).

### Ruff Rules Enforced
- `E`: pycodestyle
- `F`: pyflakes
- `UP`: pyupgrade
- `B`: bugbear
- `I`: isort
- `G`: logging format

---

## 8. Simple Code Reviewer

**Model**: Haiku
**File**: `.claude/agents/simple-code-reviewer.md`

### Purpose
Lightweight code reviewer for quick reviews of the verl codebase.

### When It Activates
Manually requested for quick code reviews.

### Example Usage
```
"Review this change for me"
"Quick code review of my actor implementation"
"Check if this follows verl conventions"
```

### Review Checklist

**Style**
- Line length ≤ 120 characters
- Proper import ordering (stdlib → third-party → verl)
- Modern Python syntax (dict[], X | Y, type[])
- Apache 2.0 license header present
- `__all__` defined for public modules

**Patterns**
- DataProto used for worker communication
- Registry pattern for extensible components
- Lazy imports for heavy dependencies
- Proper dispatch decorator usage
- GPU memory cleanup in appropriate places

**Safety**
- No GPU-CPU sync in hot paths (`.item()`, `.tolist()`)
- Proper error handling for optional dependencies
- Tensor shapes documented with comments
- No hardcoded device references (use `verl.utils.device`)

**Distributed**
- Collectives called by all ranks consistently
- Process groups passed explicitly
- No module-level global state for distributed

**Testing**
- New functions have corresponding tests
- GPU tests skip gracefully when no GPU
- Edge cases covered (empty input, NaN, boundaries)

### Output Format
```
## Code Review: [file/feature]

### Issues
- 🔴 Critical: ...
- 🟡 Warning: ...
- 🔵 Style: ...

### Suggestions
- ...

### Positive
- ...
```

---

## Agent Usage Tips

### 1. Let Agents Activate Automatically
Simply ask your question naturally. The system will activate the appropriate agent based on the topic.

### 2. Be Specific
Instead of: "How does training work?"
Try: "How does the Ray-based training loop orchestrate worker communication?"

### 3. Reference Specific Files
"How does `dp_actor.py` implement G-OPD's reward scaling?"

### 4. Ask for Examples
"Show me an example of adding a new advantage estimator"

### 5. Combine Agent Expertise
Complex questions may activate multiple agents. For example:
- "How do I add a new algorithm?" → Planner + Algorithm Expert
- "Debug FSDP checkpoint loading" → FSDP Engine Expert + Code Verifier

---

## Agent Interaction Flow

```
User Question
    ↓
System identifies relevant agent(s)
    ↓
Agent activates and reads relevant source files
    ↓
Agent provides expert guidance with:
    - Specific file references
    - Code examples
    - Configuration examples
    - Best practices
    ↓
User implements changes
    ↓
Code Verifier automatically checks quality
```
