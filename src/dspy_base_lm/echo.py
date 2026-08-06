"""A deterministic provider and LM for learning and smoke verification."""

import dspy
from typing_extensions import override

from dspy_base_lm.lm import CustomLM
from dspy_base_lm.provider import LMProvider


class EchoProvider(LMProvider):
    """Return the latest user message without using an external transport."""

    @override
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        """Echo the latest user text synchronously."""
        _ = num_retries
        return dspy.LMResponse.from_text(self._latest_user_text(request), model=request.model)

    @override
    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        """Echo the latest user text asynchronously."""
        return self.complete(request, num_retries=num_retries)

    @staticmethod
    def _latest_user_text(request: dspy.LMRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user" and message.text is not None:
                return message.text
        return ""


class EchoLM(CustomLM):
    """A reconstructable custom LM that explicitly selects EchoProvider."""

    @override
    def infer_provider(self) -> LMProvider:
        """Return the deterministic provider used by this runnable example."""
        return EchoProvider()
