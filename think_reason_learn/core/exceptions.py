"""Exceptions for the core module."""


class DataError(Exception):
    """Data not in the expected format."""


class LLMError(Exception):
    """LLM failed to respond."""


class CorruptionError(Exception):
    """Internal state corruption detected. A model's state is corrupted."""


class JevAuthError(LLMError):
    """Typesafe or OpenRouter rejected the Jev API key (HTTP 401 or 403)."""


class JevCostCapError(LLMError):
    """A Jev run would pass, or has passed, its cost cap (``max_cost_usd``)."""


class MissingAPIKeyError(ValueError):
    """A model was chosen whose API key is not set."""
