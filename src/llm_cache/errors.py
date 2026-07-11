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
