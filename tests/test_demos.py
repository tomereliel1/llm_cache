from llm_cache.demos.grpc_vector_store_orchestrator_demo import (
    local_vector_store_target,
    run_demo,
)
from llm_cache.demos.monolit_run import build_stub_demo_config


def test_monolit_run_demo_uses_stub_vector_store() -> None:
    config = build_stub_demo_config()

    assert config.vector_store.provider == "vector-store-miss-stub"
    assert config.llm.provider == "ollama"


def test_grpc_vector_store_orchestrator_demo_uses_cache() -> None:
    with local_vector_store_target() as target:
        first_hit, second_hit, llm_calls = run_demo("hello cache", target)

    assert first_hit is False
    assert second_hit is True
    assert llm_calls == 1
