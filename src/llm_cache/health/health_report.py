from collections.abc import Sequence

from llm_cache.health.health_check_result import HealthCheckResult
from llm_cache.health.health_runner import all_healthy


def format_health_report(results: Sequence[HealthCheckResult]) -> str:
    lines: list[str] = []

    if all_healthy(results):
        lines.append("Setup check passed.")
    else:
        lines.append("Setup check failed.")

    lines.append("")

    for result in results:
        status = "OK" if result.healthy else "FAIL"
        lines.append(f"[{status}] {result.name}: {result.message}")

        if result.details:
            lines.append(f"       {result.details}")

    return "\n".join(lines)
