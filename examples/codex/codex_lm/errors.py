"""Map Codex backend failures onto DSPy's typed LM errors."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import dspy

from .json_values import as_json_dict, get_dict, get_str

if TYPE_CHECKING:
    import httpx

PROVIDER_NAME = "codex"


def error_from_http_status(
    status: int,
    body: str,
    *,
    model: str,
    retry_after: float | None = None,
) -> dspy.LMError:
    """Map one non-OK HTTP response to the most specific DSPy error."""
    message, code = _message_and_code(body)
    error_type = _classify(status, code, message.lower())
    return error_type(
        message=message,
        model=model,
        provider=PROVIDER_NAME,
        provider_code=code,
        status=status,
        retry_after=retry_after,
    )


def error_from_failed_event(event: dict[str, Any], *, model: str) -> dspy.LMError:
    """Map a ``response.failed`` or ``error`` stream event to a DSPy error."""
    response = get_dict(event, "response")
    error = (get_dict(response, "error") if response is not None else None) or get_dict(
        event, "error"
    )
    source = error if error is not None else event
    message = get_str(source, "message") or "The Codex backend reported an unknown error."
    code = get_str(source, "code")
    error_type = _classify(None, code, message.lower())
    return error_type(message=message, model=model, provider=PROVIDER_NAME, provider_code=code)


def retry_after_seconds(headers: httpx.Headers) -> float | None:
    """Read the provider-suggested retry delay from response headers."""
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return max(0.0, float(retry_after_ms) / 1000.0)
        except ValueError:
            pass
    retry_after = headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        pass
    try:
        date = parsedate_to_datetime(retry_after)
    except ValueError:
        return None
    return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())


def _message_and_code(body: str) -> tuple[str, str | None]:
    """Extract the human message and error code from an error response body."""
    text = body.strip()
    fallback = text or "The Codex backend returned an error without a body."
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError:
        return fallback, None
    detail = as_json_dict(parsed)
    if detail is None:
        return fallback, None
    error = get_dict(detail, "error")
    if error is not None:
        return get_str(error, "message") or fallback, get_str(error, "code")
    return get_str(detail, "detail") or fallback, get_str(detail, "code")


def _classify(status: int | None, code: str | None, lowered: str) -> type[dspy.LMError]:
    """Choose the most specific DSPy error type for a backend failure."""
    lowered_code = (code or "").lower()
    error_type: type[dspy.LMError]
    if "context window" in lowered or "context_length" in lowered_code:
        error_type = dspy.ContextWindowExceededError
    elif (
        status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}
        or "unauthorized" in lowered
        or "token expired" in lowered
    ):
        error_type = dspy.LMAuthError
    elif (
        status == HTTPStatus.PAYMENT_REQUIRED
        or "billing" in lowered
        or lowered_code == "usage_not_included"
    ):
        error_type = dspy.LMBillingError
    elif (
        status == HTTPStatus.TOO_MANY_REQUESTS
        or "rate limit" in lowered
        or "usage limit" in lowered
        or lowered_code in {"rate_limit_exceeded", "usage_limit_reached"}
    ):
        error_type = dspy.LMRateLimitError
    elif (status is not None and status >= HTTPStatus.INTERNAL_SERVER_ERROR) or (
        "server_error" in lowered_code
    ):
        error_type = dspy.LMServerError
    elif "model is not supported" in lowered or "model_not_found" in lowered_code:
        error_type = dspy.LMUnsupportedModelError
    elif status is not None:
        error_type = dspy.LMInvalidRequestError
    else:
        error_type = dspy.LMProviderError
    return error_type
