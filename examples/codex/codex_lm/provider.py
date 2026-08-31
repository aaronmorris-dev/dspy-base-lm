"""The DSPy provider for ChatGPT/Codex subscriptions."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import dspy
from typing_extensions import override

from dspy_base_lm import LMProvider

from .auth import CodexAuth
from .response import build_lm_response
from .translate import build_request_body
from .transport import DEFAULT_BASE_URL, ResponsesTransport

_MAX_BACKOFF_SECONDS = 30.0


class CodexProvider(LMProvider):
    """Complete DSPy requests against a ChatGPT/Codex subscription.

    The provider owns the whole backend integration: credentials come from the
    Codex CLI's ``codex login`` session, each request is one streamed call to
    the ChatGPT backend Responses endpoint, and failures map to typed
    ``dspy.LMError`` types. Transient failures are retried with a capped delay
    that honors the backend's suggested retry timing.
    """

    def __init__(
        self,
        *,
        codex_home: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        read_timeout: float = 600.0,
    ) -> None:
        """Wire credentials and transport; nothing connects until the first call."""
        super().__init__()
        self._transport = ResponsesTransport(
            CodexAuth(codex_home),
            base_url=base_url,
            read_timeout=read_timeout,
        )

    @override
    def supports_function_calling(self, model: str) -> bool:
        """Codex models take native function tools."""
        _ = model
        return True

    @override
    def supports_reasoning(self, model: str) -> bool:
        """Codex models reason natively; effort and summary controls apply."""
        _ = model
        return True

    @override
    def supports_response_schema(self, model: str) -> bool:
        """JSON schemas are enforced natively through the ``text.format`` control."""
        _ = model
        return True

    @override
    def supported_params(self, model: str) -> frozenset[str]:
        """Return the request config fields the Responses endpoint honors."""
        _ = model
        return frozenset({"reasoning", "response_format", "tool_choice", "prompt_cache"})

    @override
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        """Complete one request synchronously, retrying transient failures."""
        body = build_request_body(request)
        for attempt in range(num_retries):
            try:
                return self._complete_once(body, request.model)
            except dspy.LMError as error:
                if not dspy.is_retryable_lm_error(error):
                    raise
                time.sleep(_retry_delay(attempt, error))
        return self._complete_once(body, request.model)

    @override
    async def acomplete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        """Complete one request asynchronously, retrying transient failures."""
        body = build_request_body(request)
        for attempt in range(num_retries):
            try:
                return await self._acomplete_once(body, request.model)
            except dspy.LMError as error:
                if not dspy.is_retryable_lm_error(error):
                    raise
                await asyncio.sleep(_retry_delay(attempt, error))
        return await self._acomplete_once(body, request.model)

    def close(self) -> None:
        """Close the synchronous HTTP client."""
        self._transport.close()

    async def aclose(self) -> None:
        """Close the asynchronous HTTP client."""
        await self._transport.aclose()

    def _complete_once(self, body: dict[str, Any], model: str) -> dspy.LMResponse:
        response = self._transport.request(body, model=model)
        return build_lm_response(response, model=model)

    async def _acomplete_once(self, body: dict[str, Any], model: str) -> dspy.LMResponse:
        response = await self._transport.arequest(body, model=model)
        return build_lm_response(response, model=model)


def _retry_delay(attempt: int, error: dspy.LMError) -> float:
    """Honor the backend's suggested delay when present, else back off exponentially."""
    delay = error.retry_after
    if delay is None:
        delay = 2.0**attempt
    return min(delay, _MAX_BACKOFF_SECONDS)
