from collections.abc import Sequence

from llm_cache.health.health_check_result import HealthCheckResult
from llm_cache.health.health_checkable import HealthCheckable


def run_health_checks(components: Sequence[HealthCheckable]) -> list[HealthCheckResult]:
    results: list[HealthCheckResult] = []

    for component in components:
        try:
            result = component.health_check()
        except Exception as error:
            result = HealthCheckResult.fail(
                name=component.__class__.__name__,
                message="Health check crashed",
                details=str(error),
            )

        results.append(result)

    return results


def all_healthy(results: Sequence[HealthCheckResult]) -> bool:
    return all(result.healthy for result in results)
