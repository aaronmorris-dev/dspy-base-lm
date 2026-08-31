"""HTTP transport for the ChatGPT backend's Codex Responses endpoint."""

from __future__ import annotations

import asyncio
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import dspy
import httpx

from .errors import (
    PROVIDER_NAME,
    error_from_failed_event,
    error_from_http_status,
    retry_after_seconds,
)
from .json_values import get_dict
from .sse import ServerSentEventParser

if TYPE_CHECKING:
    from .auth import CodexAuth, CodexTokens

DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex/responses"
_ORIGINATOR = "dspy_base_lm"
_CONNECT_TIMEOUT_SECONDS = 30.0
_FINAL_EVENT_TYPES = frozenset({"response.completed", "response.incomplete"})
_FAILURE_EVENT_TYPES = frozenset({"response.failed", "error"})
_ITEM_DONE_EVENT_TYPE = "response.output_item.done"


class ResponsesTransport:
    """One HTTP client pair for streaming Responses requests.

    Each request is one POST whose Server-Sent Events are consumed until the
    final response object arrives. Rejected credentials are refreshed once and
    the request is retried with the new tokens.
    """

    def __init__(
        self,
        auth: CodexAuth,
        *,
        base_url: str = DEFAULT_BASE_URL,
        read_timeout: float = 600.0,
    ) -> None:
        """Create clients for ``base_url`` with ``read_timeout`` between stream chunks."""
        self._auth = auth
        self._url = base_url
        timeout = httpx.Timeout(_CONNECT_TIMEOUT_SECONDS, read=read_timeout)
        self._client = httpx.Client(timeout=timeout)
        self._async_client = httpx.AsyncClient(timeout=timeout)

    def request(self, body: dict[str, Any], *, model: str) -> dict[str, Any]:
        """POST one request body and return the final response object."""
        tokens = self._auth.tokens()
        try:
            return self._send(body, tokens, model=model)
        except dspy.LMAuthError as error:
            if error.status != HTTPStatus.UNAUTHORIZED:
                raise
            return self._send(body, self._auth.refresh(tokens.access_token), model=model)

    async def arequest(self, body: dict[str, Any], *, model: str) -> dict[str, Any]:
        """POST one request body and return the final response object."""
        tokens = self._auth.tokens()
        try:
            return await self._asend(body, tokens, model=model)
        except dspy.LMAuthError as error:
            if error.status != HTTPStatus.UNAUTHORIZED:
                raise
            refreshed = await asyncio.to_thread(self._auth.refresh, tokens.access_token)
            return await self._asend(body, refreshed, model=model)

    def close(self) -> None:
        """Close the synchronous HTTP client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the asynchronous HTTP client."""
        await self._async_client.aclose()

    def _send(
        self,
        body: dict[str, Any],
        tokens: CodexTokens,
        *,
        model: str,
    ) -> dict[str, Any]:
        try:
            with self._client.stream(
                "POST",
                self._url,
                json=body,
                headers=_headers(tokens),
            ) as response:
                if response.status_code != HTTPStatus.OK:
                    raise error_from_http_status(
                        response.status_code,
                        response.read().decode("utf-8", errors="replace"),
                        model=model,
                        retry_after=retry_after_seconds(response.headers),
                    )
                parser = ServerSentEventParser()
                accumulator = _StreamAccumulator(model)
                for line in response.iter_lines():
                    accumulator.note(parser.feed(line))
                accumulator.note(parser.flush())
        except httpx.TimeoutException as error:
            raise _timeout_error(error, model=model) from error
        except httpx.HTTPError as error:
            raise _transport_error(error, model=model) from error
        return accumulator.final_response()

    async def _asend(
        self,
        body: dict[str, Any],
        tokens: CodexTokens,
        *,
        model: str,
    ) -> dict[str, Any]:
        try:
            async with self._async_client.stream(
                "POST",
                self._url,
                json=body,
                headers=_headers(tokens),
            ) as response:
                if response.status_code != HTTPStatus.OK:
                    raise error_from_http_status(
                        response.status_code,
                        (await response.aread()).decode("utf-8", errors="replace"),
                        model=model,
                        retry_after=retry_after_seconds(response.headers),
                    )
                parser = ServerSentEventParser()
                accumulator = _StreamAccumulator(model)
                async for line in response.aiter_lines():
                    accumulator.note(parser.feed(line))
                accumulator.note(parser.flush())
        except httpx.TimeoutException as error:
            raise _timeout_error(error, model=model) from error
        except httpx.HTTPError as error:
            raise _transport_error(error, model=model) from error
        return accumulator.final_response()


def _headers(tokens: CodexTokens) -> dict[str, str]:
    """Build per-request headers carrying credentials and a request id."""
    request_id = str(uuid.uuid4())
    return {
        "authorization": f"Bearer {tokens.access_token}",
        "chatgpt-account-id": tokens.account_id,
        "originator": _ORIGINATOR,
        "user-agent": f"{_ORIGINATOR}_codex_example",
        "openai-beta": "responses=experimental",
        "accept": "text/event-stream",
        "session-id": request_id,
        "x-client-request-id": request_id,
    }


class _StreamAccumulator:
    """Collect one request's streamed output items and final response object.

    The backend omits ``output`` from the final ``response.completed`` payload
    and streams each output item through ``response.output_item.done`` events
    instead, so the collected items are folded back into the final response.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self._items: list[dict[str, Any]] = []
        self._final: dict[str, Any] | None = None

    def note(self, event: dict[str, Any] | None) -> None:
        """Record one parsed stream event; raise on failure events."""
        if event is None:
            return
        event_type = event.get("type")
        if event_type in _FAILURE_EVENT_TYPES:
            raise error_from_failed_event(event, model=self._model)
        if event_type == _ITEM_DONE_EVENT_TYPE:
            item = get_dict(event, "item")
            if item is not None:
                self._items.append(item)
        elif event_type in _FINAL_EVENT_TYPES:
            self._final = get_dict(event, "response")

    def final_response(self) -> dict[str, Any]:
        """Return the final response with streamed output items folded in."""
        if self._final is None:
            message = "The Codex backend stream ended without a final response."
            raise dspy.LMTransportError(message, model=self._model, provider=PROVIDER_NAME)
        if not self._final.get("output"):
            self._final["output"] = self._items
        return self._final


def _timeout_error(error: httpx.TimeoutException, *, model: str) -> dspy.LMTimeoutError:
    message = f"The Codex backend request timed out: {error!r}."
    return dspy.LMTimeoutError(message, model=model, provider=PROVIDER_NAME)


def _transport_error(error: httpx.HTTPError, *, model: str) -> dspy.LMTransportError:
    message = f"The Codex backend request failed in transit: {error!r}."
    return dspy.LMTransportError(message, model=model, provider=PROVIDER_NAME)
