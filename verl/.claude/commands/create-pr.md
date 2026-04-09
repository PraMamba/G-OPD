# Create Pull Request

Full PR workflow: fetch, rebase, push, create/update PR with intelligent messages.

## Arguments
- None: Create PR for current branch
- `--draft`: Create as draft PR

## Workflow

### Step 1: Pre-flight Checks
```bash
# Ensure we're not on main
git branch --show-current

# Check for uncommitted changes
git status --short

# Fetch latest
git fetch origin
```

### Step 2: Rebase on Main
```bash
git rebase origin/main
```
If conflicts, report and stop.

### Step 3: Analyze Changes
```bash
# Get full diff against main
git log origin/main..HEAD --oneline
git diff origin/main..HEAD --stat
git diff origin/main..HEAD
```

### Step 4: Generate PR Title and Body

**Title format**: `<type>(<scope>): <description>` (under 70 chars)

**Body template**:
```markdown
## Summary
<1-3 bullet points describing the change>

## Changes
| File | Description |
|------|-------------|
| ... | ... |

## Testing
- [ ] Ruff lint passes
- [ ] Relevant unit tests pass
- [ ] Manual testing (if applicable)

## Configuration
<Any new config parameters or CLI overrides>
```

### Step 5: Push and Create PR
```bash
git push -u origin <branch-name>

gh pr create \
  --title "<title>" \
  --body "<body>" \
  --base main
```

### Step 6: Report
Output the PR URL and a summary.
