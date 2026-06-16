from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    command: list[str]
    return_code: int
    output: str

    @property
    def passed(self) -> bool:
        return self.return_code == 0


@dataclass(frozen=True)
class FileSnapshot:
    relative_path: Path
    content: str


def run_command(name: str, command: list[str], project_root: Path) -> CheckResult:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(project_root / ".uv-cache"))

    completed = subprocess.run(
        command,
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )

    return CheckResult(
        name=name,
        command=command,
        return_code=completed.returncode,
        output=output,
    )


def iter_python_files(project_root: Path) -> list[Path]:
    python_files: list[Path] = []

    for path in project_root.rglob("*.py"):
        relative_parts = path.relative_to(project_root).parts
        if any(part in EXCLUDED_DIRS for part in relative_parts):
            continue
        python_files.append(path)

    return sorted(python_files)


def snapshot_python_files(project_root: Path) -> dict[Path, FileSnapshot]:
    snapshots: dict[Path, FileSnapshot] = {}

    for path in iter_python_files(project_root):
        relative_path = path.relative_to(project_root)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        snapshots[relative_path] = FileSnapshot(
            relative_path=relative_path,
            content=content,
        )

    return snapshots


def diff_snapshots(
    before: dict[Path, FileSnapshot],
    after: dict[Path, FileSnapshot],
) -> tuple[list[Path], str]:
    changed_paths: list[Path] = []
    diff_lines: list[str] = []

    for relative_path in sorted(before.keys() | after.keys()):
        before_content = before.get(relative_path)
        after_content = after.get(relative_path)

        before_text = before_content.content if before_content else ""
        after_text = after_content.content if after_content else ""

        if before_text == after_text:
            continue

        changed_paths.append(relative_path)
        diff_lines.extend(
            difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"before/{relative_path}",
                tofile=f"after/{relative_path}",
            )
        )

        if diff_lines and not diff_lines[-1].endswith("\n"):
            diff_lines[-1] += "\n"

    return changed_paths, "".join(diff_lines)


def check_python_compilation(project_root: Path) -> CheckResult:
    errors: list[str] = []
    python_files = iter_python_files(project_root)

    for path in python_files:
        relative_path = path.relative_to(project_root)
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(relative_path), "exec")
        except SyntaxError as error:
            location = f"{relative_path}:{error.lineno}:{error.offset}"
            errors.append(f"{location}: {error.msg}")
        except UnicodeDecodeError as error:
            errors.append(f"{relative_path}: could not decode as UTF-8: {error}")

    if errors:
        output = "\n".join(errors)
        return_code = 1
    else:
        output = f"Compiled {len(python_files)} Python files in memory."
        return_code = 0

    return CheckResult(
        name="Python compilation",
        command=["python", "compile(source, filename, 'exec')"],
        return_code=return_code,
        output=output,
    )


def format_command(command: list[str]) -> str:
    return " ".join(command)


def build_report(results: list[CheckResult]) -> str:
    all_passed = all(result.passed for result in results)
    status = "PASS" if all_passed else "FAIL"
    lines = [
        "# Project Quality Check Report",
        "",
        f"Overall status: {status}",
        "",
        "## Summary",
        "",
    ]

    for result in results:
        check_status = "PASS" if result.passed else "FAIL"
        lines.append(f"- {result.name}: {check_status}")

    for result in results:
        check_status = "PASS" if result.passed else "FAIL"
        lines.extend(
            [
                "",
                f"## {result.name}",
                "",
                f"Status: {check_status}",
                f"Command: `{format_command(result.command)}`",
                "",
                "```text",
                result.output or "No output.",
                "```",
            ]
        )

    return "\n".join(lines) + "\n"


def append_results_section(
    lines: list[str],
    heading: str,
    results: list[CheckResult],
) -> None:
    lines.extend(["", f"## {heading}", ""])

    for result in results:
        check_status = "PASS" if result.passed else "FAIL"
        lines.append(f"- {result.name}: {check_status}")

    for result in results:
        check_status = "PASS" if result.passed else "FAIL"
        lines.extend(
            [
                "",
                f"### {result.name}",
                "",
                f"Status: {check_status}",
                f"Command: `{format_command(result.command)}`",
                "",
                "```text",
                result.output or "No output.",
                "```",
            ]
        )


def build_fix_report(
    before_results: list[CheckResult],
    fix_results: list[CheckResult],
    after_results: list[CheckResult],
    changed_paths: list[Path],
    diff: str,
) -> str:
    all_passed = all(result.passed for result in after_results)
    status = "PASS" if all_passed else "FAIL"
    lines = [
        "# Project Quality Check Report",
        "",
        f"Overall status: {status}",
        "",
        "## Summary",
        "",
        f"- Initial Ruff lint: {'PASS' if before_results[0].passed else 'FAIL'}",
        f"- Initial Ruff format: {'PASS' if before_results[1].passed else 'FAIL'}",
        f"- Initial Python compilation: {'PASS' if before_results[2].passed else 'FAIL'}",
        f"- Ruff format fix: {'PASS' if fix_results[0].passed else 'FAIL'}",
        f"- Ruff lint fix: {'PASS' if fix_results[1].passed else 'FAIL'}",
        f"- Final Ruff lint: {'PASS' if after_results[0].passed else 'FAIL'}",
        f"- Final Ruff format: {'PASS' if after_results[1].passed else 'FAIL'}",
        f"- Final Python compilation: {'PASS' if after_results[2].passed else 'FAIL'}",
        f"- Files changed by fixes: {len(changed_paths)}",
    ]

    append_results_section(lines, "Initial Check Results", before_results)
    append_results_section(lines, "Automatic Fix Commands", fix_results)

    lines.extend(["", "## Changes Made By Automatic Fixes", ""])
    if changed_paths:
        lines.append("Changed files:")
        lines.extend(f"- {path}" for path in changed_paths)
        lines.extend(["", "```diff", diff.rstrip(), "```"])
    else:
        lines.append("No file content changes detected.")

    append_results_section(lines, "Final Check Results", after_results)

    return "\n".join(lines) + "\n"


def write_report(project_root: Path, report: str, report_path: Path) -> Path:
    destination = report_path
    if not destination.is_absolute():
        destination = project_root / destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return destination


def run_checks(project_root: Path) -> list[CheckResult]:
    return [
        run_command("Ruff lint", ["uv", "run", "ruff", "check", "."], project_root),
        run_command(
            "Ruff format",
            ["uv", "run", "ruff", "format", "--check", "."],
            project_root,
        ),
        check_python_compilation(project_root),
    ]


def run_automatic_fixes(project_root: Path) -> list[CheckResult]:
    return [
        run_command("Ruff format fix", ["uv", "run", "ruff", "format", "."], project_root),
        run_command(
            "Ruff lint fix",
            ["uv", "run", "ruff", "check", ".", "--fix"],
            project_root,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project lint, format, and compile checks.")
    parser.add_argument(
        "--report-path",
        default=".codex/reports/project_quality_checks.md",
        type=Path,
        help="Path to write the Markdown report.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Run Ruff format and Ruff lint auto-fixes before writing the final check report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()

    if args.fix:
        before_results = run_checks(project_root)
        before_snapshot = snapshot_python_files(project_root)
        fix_results = run_automatic_fixes(project_root)
        after_snapshot = snapshot_python_files(project_root)
        results = run_checks(project_root)
        changed_paths, diff = diff_snapshots(before_snapshot, after_snapshot)

        report = build_fix_report(
            before_results=before_results,
            fix_results=fix_results,
            after_results=results,
            changed_paths=changed_paths,
            diff=diff,
        )
    else:
        results = run_checks(project_root)
        report = build_report(results)

    destination = write_report(project_root, report, args.report_path)

    print(report)
    print(f"Report written to: {destination}")

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
