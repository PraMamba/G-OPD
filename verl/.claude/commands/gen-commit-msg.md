# Generate Commit Message

Generate a clear, conventional commit message based on staged changes.

## Arguments
None required. Analyzes current git state.

## Steps

1. **Analyze changes**:
   ```bash
   git diff --cached --stat
   git diff --cached
   ```

2. **Categorize change type**:
   - `feat`: New feature
   - `fix`: Bug fix
   - `refactor`: Code refactoring
   - `test`: Adding or modifying tests
   - `docs`: Documentation changes
   - `chore`: Build/config changes
   - `perf`: Performance improvement

3. **Identify scope** from the changed files:
   - `actor`: Actor worker changes
   - `critic`: Critic worker changes
   - `rollout`: Rollout engine changes
   - `trainer`: Training loop changes
   - `algo`: Algorithm changes (core_algos, policy loss)
   - `reward`: Reward function changes
   - `config`: Configuration changes
   - `utils`: Utility changes
   - `g-opd`: G-OPD/ExOPD specific changes
   - `protocol`: DataProto changes
   - `sharding`: Weight sync changes

4. **Generate message** in format:
   ```
   type(scope): concise description

   - Detail 1
   - Detail 2
   ```

## Examples
```
feat(algo): add OPO advantage estimator

- Register OPO via @register_adv_est decorator
- Implement length-weighted baseline subtraction
- Add token-level and sequence-level variants
```

```
fix(g-opd): correct multi-teacher KL routing for code teacher

- Fix base_ref_log_prob selection for code teacher samples
- Ensure opd_teacher field correctly routes to appropriate log probs
```
