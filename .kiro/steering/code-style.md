# Code Style and Conventions

## License Headers

All Python files must include the Apache 2.0 license header:

```python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

For files with third-party contributions, include additional copyright lines before the Bytedance copyright.

## Import Organization

Follow this import order (enforced by ruff with isort):
1. Standard library imports
2. Third-party imports (torch, numpy, ray, etc.)
3. Local imports from verl package

Use absolute imports: `from verl.protocol import DataProto` not relative imports.

Group imports logically and use `from X import Y` for commonly used items.

## Module Structure

- Use `__all__` to explicitly define public API in `__init__.py` files
- Place module docstrings immediately after license header
- Abstract base classes go in `base.py` files
- Concrete implementations in separate files (e.g., `dp_actor.py`, `dp_critic.py`)

## Naming Conventions

- Classes: PascalCase (e.g., `DataParallelPPOActor`, `BasePPOCritic`)
- Functions/methods: snake_case (e.g., `compute_log_prob`, `update_policy`)
- Constants: UPPER_SNAKE_CASE (e.g., `POLICY_LOSS_REGISTRY`)
- Private methods: prefix with `_` (e.g., `_forward_micro_batch`, `_optimizer_step`)
- Config classes: suffix with `Config` (e.g., `ActorConfig`, `AlgoConfig`)

## Code Formatting

- Line length: 120 characters (ruff configured)
- Use type hints for function signatures
- Docstrings: Google style with Args/Returns sections
- Use f-strings for string formatting
- Avoid lambda assignments (ruff rule E731 ignored)

## Linting Rules

Ruff is configured with these rules:
- E (pycodestyle errors)
- F (Pyflakes)
- UP (pyupgrade)
- B (flake8-bugbear)
- I (isort)
- G (logging format)

Ignored rules:
- F405, F403 (star imports)
- E731 (lambda assignments)
- B007 (unused loop variables)
- UP032 (f-string format)
- G004 (f-strings in logging)
