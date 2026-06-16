from typing import Protocol

from llm_cache.health.health_check_result import HealthCheckResult


class HealthCheckable(Protocol):
    def health_check(self) -> HealthCheckResult: ...
