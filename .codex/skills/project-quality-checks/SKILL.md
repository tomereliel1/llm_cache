---
name: project-quality-checks
description: Run project-local Ruff lint, Ruff format, and Python compilation checks, produce a concise issue report, ask the user once before fixing anything, then apply approved fixes and rerun checks until verified. Use only when explicitly invoked as $project-quality-checks or when the user specifically names project-quality-checks for the llm_cache project.
---

# Project Quality Checks

## Workflow

Use this skill only after explicit invocation. Do not trigger it for ordinary coding tasks.

1. Run the report script from the project root:

```bash
uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py
```

2. Report the results to the user:
   - State whether lint, format, and compilation passed.
   - Include each failing command's issue output.
   - Do not fix anything yet.

3. Ask for permission to fix the reported issues.

4. After permission, run the automatic fix command:

```bash
uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py --fix
```

This command applies fixes in this order:
   - Run `uv run ruff format .`.
   - Run `uv run ruff check . --fix`.
   - Rerun the full lint, format, and compilation report.
   - Write one final report that includes the initial check output, automatic fix command output, the files changed by fixes, a unified before/after diff of those file changes, and the final check output.

5. If Ruff or compilation issues remain after automatic fixes, fix them manually with focused code edits.

6. Rerun the report script after any manual edits.

7. Keep fixing and rerunning checks under the original fix approval until all initially reported issues are resolved. Do not ask for additional permission to run project quality commands after the user has approved the initial fix pass. If new unrelated issues appear or a fix requires a destructive action, stop and ask the user. If all checks pass, tell the user it is fixed and include the final report path.

## Report Script

The script writes a Markdown report to `.codex/reports/project_quality_checks.md` and prints the same report to stdout. Use the report as the source of truth for the user-facing summary.

When running with `--fix`, the report must preserve evidence of what changed. It snapshots Python files before automatic fixes, snapshots them again after automatic fixes, and includes a unified diff under `Changes Made By Automatic Fixes`. This makes output like `Would reformat: path/to/file.py` traceable to the exact lines that were changed.

Compilation is checked by parsing and compiling every tracked Python source file in memory. This catches syntax errors without creating `__pycache__` files.

## Permission Rule

Never run formatting, automatic fixes, or manual edits until the user has approved fixing the issues found in the initial report. That approval covers all non-destructive quality-check commands and focused edits needed to resolve those issues, including repeated `uv run python .codex/skills/project-quality-checks/scripts/check_project_quality.py --fix` and report reruns.
