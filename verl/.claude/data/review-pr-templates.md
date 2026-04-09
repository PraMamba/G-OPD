# PR Review Templates

## Template: Algorithm Change
```markdown
### Algorithm Review: <area>
**Risk**: CRITICAL/HIGH
**Files**: <list>

#### Correctness
- [ ] Mathematical formulation matches paper/spec
- [ ] Gradient computation is correct
- [ ] Loss aggregation mode is appropriate
- [ ] Advantage normalization is consistent

#### Numerical Stability
- [ ] No division by zero
- [ ] Log probabilities handled correctly (logsumexp)
- [ ] Clipping values are reasonable
- [ ] NaN/inf checks where needed

#### Performance
- [ ] No unnecessary GPU-CPU sync
- [ ] Efficient tensor operations (batch ops over loops)
- [ ] Memory-efficient (chunking for large tensors)
```

## Template: Worker Change
```markdown
### Worker Review: <area>
**Risk**: HIGH/MEDIUM
**Files**: <list>

#### DataProto Contract
- [ ] Input keys documented/validated
- [ ] Output keys match expectations
- [ ] Split/chunk used correctly for micro-batching
- [ ] Meta info passed through correctly

#### Distributed Correctness
- [ ] Dispatch decorator is appropriate
- [ ] All ranks call same collectives
- [ ] No deadlocks in conditional branches
- [ ] Process groups passed explicitly

#### Resource Management
- [ ] GPU memory cleaned after use
- [ ] vLLM/SGLang memory freed via ShardingManager
- [ ] Gradient accumulation cleared between steps
```

## Template: Configuration Change
```markdown
### Config Review: <area>
**Risk**: MEDIUM/LOW
**Files**: <list>

#### Compatibility
- [ ] New fields have defaults (backward compatible)
- [ ] Removed fields have deprecation warnings
- [ ] YAML comments above fields
- [ ] CLI help text for exposed parameters

#### Validation
- [ ] __post_init__ validates constraints
- [ ] Error messages are clear
- [ ] Edge cases handled
```

## Template: General Code
```markdown
### Code Review: <area>
**Risk**: <level>
**Files**: <list>

#### Style
- [ ] Follows verl code conventions
- [ ] Line length ≤ 120
- [ ] Import order correct
- [ ] License header present

#### Logic
- [ ] Correct behavior
- [ ] Edge cases handled
- [ ] Error handling appropriate

#### Testing
- [ ] Tests added/updated
- [ ] Coverage adequate
```
