from collections.abc import Iterable


def print_server_started(
    server_name: str,
    address: str,
    details: Iterable[tuple[str, str]] = (),
) -> None:
    print()
    print(f"[READY] {server_name} gRPC server started successfully")
    print(f"  Address: {address}")
    for label, value in details:
        print(f"  {label}: {value}")
    print()
