class ProviderUnavailableError(RuntimeError):
    """Raised when an orchestrator dependency cannot be reached."""

    def __init__(self, provider: str, target: str, details: str) -> None:
        super().__init__(
            f"{provider} service is unavailable. Check that it is running and "
            "reachable, then try again."
        )
        self.technical_details = (
            f"Provider: {provider}\ngRPC status: UNAVAILABLE\nTarget: {target}\nDetails: {details}"
        )


class ProviderTimeoutError(RuntimeError):
    """Raised when an orchestrator dependency does not answer before the deadline."""

    def __init__(
        self,
        provider: str,
        target: str,
        details: str,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            f"{provider} service did not respond within {timeout_seconds:g} seconds. "
            "Increase the provider timeout using --provider-timeout-seconds or the "
            "provider_timeout_seconds JSON field, then try again."
        )
        self.technical_details = (
            f"Provider: {provider}\n"
            "gRPC status: DEADLINE_EXCEEDED\n"
            f"Target: {target}\n"
            f"Timeout seconds: {timeout_seconds:g}\n"
            f"Details: {details}"
        )
