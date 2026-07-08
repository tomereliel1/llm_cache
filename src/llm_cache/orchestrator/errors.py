class OrchestratorClientError(RuntimeError):
    """Public client error with optional diagnostic information."""

    def __init__(self, message: str, *, technical_details: str | None = None) -> None:
        super().__init__(message)
        self.technical_details = technical_details

