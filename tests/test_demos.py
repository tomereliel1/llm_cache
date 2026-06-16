from llm_cache.demos.monolit_run import build_stub_demo_config


def test_monolit_run_demo_uses_stub_vector_store() -> None:
    config = build_stub_demo_config()

    assert config.vector_store.provider == "vector-store-miss-stub"
    assert config.llm.provider == "ollama"
