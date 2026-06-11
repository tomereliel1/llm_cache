import pytest

from llm_cache.health import (
    HealthCheckResult,
    all_healthy,
    format_health_report,
    run_health_checks,
    run_setup_check_and_exit,
)


class HealthyComponent:
    def health_check(self) -> HealthCheckResult:
        return HealthCheckResult.ok("fake", "ready")


class FailingComponent:
    def health_check(self) -> HealthCheckResult:
        return HealthCheckResult.fail("fake", "not ready", details="missing setup")


class CrashingComponent:
    def health_check(self) -> HealthCheckResult:
        raise RuntimeError("boom")


def test_health_check_result_ok_is_healthy() -> None:
    result = HealthCheckResult.ok("fake", "ready")

    assert result.healthy is True
    assert result.details is None


def test_health_check_result_fail_is_unhealthy() -> None:
    result = HealthCheckResult.fail("fake", "not ready", details="missing setup")

    assert result.healthy is False
    assert result.details == "missing setup"


def test_run_health_checks_collects_healthy_result() -> None:
    results = run_health_checks([HealthyComponent()])

    assert results == [HealthCheckResult.ok("fake", "ready")]
    assert all_healthy(results)


def test_run_health_checks_collects_failing_result() -> None:
    results = run_health_checks([FailingComponent()])

    assert len(results) == 1
    assert results[0].healthy is False
    assert results[0].message == "not ready"


def test_run_health_checks_reports_crashing_health_check() -> None:
    results = run_health_checks([CrashingComponent()])

    assert len(results) == 1
    assert results[0].healthy is False
    assert results[0].name == "CrashingComponent"
    assert results[0].message == "Health check crashed"
    assert results[0].details == "boom"


def test_format_health_report_for_passing_results() -> None:
    report = format_health_report([HealthCheckResult.ok("fake", "ready")])

    assert "Setup check passed." in report
    assert "[OK] fake: ready" in report


def test_format_health_report_for_failing_results_includes_details() -> None:
    report = format_health_report(
        [
            HealthCheckResult.fail("fake", "not ready", details="missing setup"),
        ]
    )

    assert "Setup check failed." in report
    assert "[FAIL] fake: not ready" in report
    assert "missing setup" in report


def test_setup_check_cli_exits_zero_for_passing_checks(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_setup_check_and_exit([HealthyComponent()])

    assert exc_info.value.code == 0
    assert "Setup check passed." in capsys.readouterr().out


def test_setup_check_cli_exits_one_for_failing_checks(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        run_setup_check_and_exit([FailingComponent()])

    assert exc_info.value.code == 1
    assert "Setup check failed." in capsys.readouterr().out
