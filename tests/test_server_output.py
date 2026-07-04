from llm_cache.server_output import print_server_started


def test_print_server_started_formats_block_with_surrounding_blank_lines(capsys) -> None:
    print_server_started(
        "Vector Store",
        "localhost:50052",
        (
            ("Provider", "in-memory"),
            ("Eviction policy", "lru"),
        ),
    )

    assert capsys.readouterr().out == (
        "\n"
        "[READY] Vector Store gRPC server started successfully\n"
        "  Address: localhost:50052\n"
        "  Provider: in-memory\n"
        "  Eviction policy: lru\n"
        "\n"
    )
