"""The provider interface for custom DSPy language models."""

from abc import ABC, abstractmethod

import dspy


class LMProvider(dspy.Provider, ABC):
    """Translate between one transport and DSPy's typed LM contract.

    Subclasses own their runtime client, request and response translation,
    retries, and mapping of known backend failures to ``dspy.LMError`` types.
    """

    def supports_function_calling(self, model: str) -> bool:
        """Whether ``model`` supports native function calling."""
        _ = model
        return False

    def supports_reasoning(self, model: str) -> bool:
        """Whether ``model`` supports native reasoning."""
        _ = model
        return False

    def supports_response_schema(self, model: str) -> bool:
        """Whether ``model`` supports native response schemas."""
        _ = model
        return False

    def supported_params(self, model: str) -> frozenset[str]:
        """Return the generation parameters supported by ``model``."""
        _ = model
        return frozenset()

    @abstractmethod
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        """Complete one normalized request synchronously."""
        raise NotImplementedError

    @abstractmethod
    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        """Complete one normalized request asynchronously."""
        raise NotImplementedError
