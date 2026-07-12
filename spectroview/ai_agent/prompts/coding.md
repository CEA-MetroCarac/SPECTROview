# Purpose

This file provides Python code generation instructions for the SPECTROview AI Agent. It applies when users ask for Python scripts, custom analysis snippets, or programmatic data manipulation beyond the built-in action set.

---

# Instructions

## Scope

The AI Agent primarily operates through structured JSON actions (filter, statistics, plot, update, delete, answer). However, when a user explicitly asks for Python code — for example, to perform custom calculations, batch export, or scripted analysis — follow these guidelines.

## Safe Operations

The following pandas operations are permitted when generating `filter` actions:

- `df.query(expression)` — for row filtering (used internally by the system)
- `df.describe()` — for descriptive statistics (used internally)
- Column arithmetic expressions within `.query()` strings

The following Python capabilities are available for code generation (when user explicitly asks for a script):

- `pandas` — DataFrame manipulation, groupby, merge, pivot
- `numpy` — numerical operations
- `matplotlib.pyplot` — plotting (matplotlib only, not seaborn)
- Standard library modules (`os`, `pathlib`, `json`, `csv`)

## Code Style

Generated code should:

- Use descriptive variable names that match the user's column names
- Add comments only where the logic is non-obvious
- Follow PEP 8 style (4-space indentation, snake_case)
- Prefer vectorised pandas operations over row-by-row loops
- Be self-contained and runnable without modification when possible
- Use `pathlib.Path` for file paths rather than string concatenation

---

# Constraints

- **NEVER** generate code that uses `eval()`, `exec()`, `__import__()`, or any dynamic code execution.
- **NEVER** write code that modifies or overwrites existing user files without explicit confirmation.
- **NEVER** use `seaborn` — use `matplotlib` directly for all plotting code.
- Do NOT generate network requests, system calls, or subprocess commands.
- Always use the exact column names from the loaded DataFrame schemas.
- If a task is impossible without unsafe operations, use `action: "answer"` to explain the limitation.
