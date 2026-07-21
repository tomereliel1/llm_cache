from pathlib import Path

from scripts.analyze_trace import (
    HitExample,
    SimulationResult,
    build_report,
    run_project_vector_store,
)


def test_build_report_includes_threshold_metrics_and_examples() -> None:
    report = build_report(
        trace_path=Path("datasets/quora_1000.h5"),
        capacity=1000,
        results=[
            SimulationResult(
                backend="precomputed-vectors",
                threshold=0.8,
                total_prompts=2,
                hits=1,
                misses=1,
                hit_rate=0.5,
                avg_hit_distance=0.1,
                best_hit_distance=0.1,
                worst_hit_distance=0.1,
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
    assert "best hit distance" in report
    assert "avg hit similarity" not in report
    assert "p50 nearest" not in report
    assert "How do I reset my password?" in report
    assert "How can I change my password?" in report


def test_project_vector_store_backend_uses_real_project_components() -> None:
    result = run_project_vector_store(
        [
            "How do I reset my password?",
            "How can I change my password?",
        ],
        threshold=0.7,
        capacity=1000,
        examples_count=1,
        embedding_provider="embedder-stub",
        embedding_model="fixed-vector",
        embedding_base_url=None,
        vector_store_provider="in-memory",
    )

    assert result.backend == "project-orchestrator:in-memory"
    assert result.total_prompts == 2
    assert result.hits == 1
    assert result.misses == 1
    assert result.avg_hit_distance == 1.0
    assert result.best_hit_distance == 1.0
    assert result.worst_hit_distance == 1.0
    assert result.examples[0].similarity == 1.0
    assert result.examples[0].cached_prompt == "How do I reset my password?"
