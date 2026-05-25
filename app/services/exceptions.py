class ConfigurationError(Exception):
    """Required application configuration is missing."""


class ProviderError(Exception):
    """Gemini request failed in a user-presentable way."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code

