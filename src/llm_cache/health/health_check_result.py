from dataclasses import dataclass


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    healthy: bool
    message: str
    details: str | None = None

    @staticmethod
    def ok(name: str, message: str) -> "HealthCheckResult":
        return HealthCheckResult(
            name=name,
            healthy=True,
            message=message,
        )

    @staticmethod
    def fail(
        name: str,
        message: str,
        details: str | None = None,
    ) -> "HealthCheckResult":
        return HealthCheckResult(
            name=name,
            healthy=False,
            message=message,
            details=details,
        )
