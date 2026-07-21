from pathlib import Path

from scripts.analyze_trace import HitExample, SimulationResult, build_report


def test_build_report_includes_threshold_metrics_and_examples() -> None:
    report = build_report(
        trace_path=Path("datasets/quora_1000.h5"),
        capacity=1000,
        results=[
            SimulationResult(
                threshold=0.8,
                total_prompts=2,
                hits=1,
                misses=1,
                hit_rate=0.5,
                avg_hit_similarity=0.9,
                avg_hit_distance=0.1,
                p50_nearest_similarity=0.9,
                p90_nearest_similarity=0.9,
                examples=[
                    HitExample(
                        prompt="How do I reset my password?",
                        cached_prompt="How can I change my password?",
                        similarity=0.9,
                        index=1,
                        cached_index=0,
                    )
                ],
            )
        ],
    )

    assert "Trace Analysis Report" in report
    assert "0.80" in report
    assert "50.0%" in report
    assert "avg hit distance" in report
    assert "How do I reset my password?" in report
    assert "How can I change my password?" in report
