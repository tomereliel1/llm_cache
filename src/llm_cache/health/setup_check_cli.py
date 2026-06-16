from collections.abc import Sequence

from llm_cache.health.health_checkable import HealthCheckable
from llm_cache.health.health_report import format_health_report
from llm_cache.health.health_runner import all_healthy, run_health_checks


def run_setup_check_and_exit(components: Sequence[HealthCheckable]) -> None:
    results = run_health_checks(components)
    print(format_health_report(results))

    if all_healthy(results):
        raise SystemExit(0)

    raise SystemExit(1)
