"""Translate Responses API results into typed DSPy responses."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import dspy
from dspy.core import (
    LMOutput,
    LMRefusalPart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMUsage,
)

from .errors import PROVIDER_NAME
from .json_values import as_json_dict, get_dict, get_int, get_list, get_str

if TYPE_CHECKING:
    from dspy.core import LMPart


def lm_response(response: dict[str, Any], *, model: str) -> dspy.LMResponse:
    """Map one final Responses object to a typed DSPy response."""
    parts = _parts(response)
    if not parts:
        message = "The Codex backend completed without any output content."
        raise dspy.LMProviderError(message, model=model, provider=PROVIDER_NAME)
    truncated = get_str(response, "status") == "incomplete"
    has_tool_call = any(isinstance(part, LMToolCallPart) for part in parts)
    finish_reason = "tool_calls" if has_tool_call else "length" if truncated else "stop"
    usage_data = get_dict(response, "usage")
    return dspy.LMResponse(
        model=model,
        outputs=[LMOutput(parts=parts, finish_reason=finish_reason, truncated=truncated)],
        usage=_usage(usage_data) if usage_data is not None else None,
        response_id=get_str(response, "id"),
        provider_response=response,
    )


def _parts(response: dict[str, Any]) -> list[LMPart]:
    """Convert the response's output items into typed LM parts."""
    parts: list[LMPart] = []
    for raw_item in get_list(response, "output") or []:
        item = as_json_dict(raw_item)
        if item is None:
            continue
        item_type = get_str(item, "type")
        if item_type == "reasoning":
            parts.extend(_thinking_parts(item))
        elif item_type == "message":
            parts.extend(_message_parts(item))
        elif item_type == "function_call":
            parts.append(_tool_call_part(item))
    return parts


def _thinking_parts(item: dict[str, Any]) -> list[LMPart]:
    """Convert a reasoning item's summary entries."""
    parts: list[LMPart] = []
    for raw_summary in get_list(item, "summary") or []:
        summary = as_json_dict(raw_summary)
        text = get_str(summary, "text") if summary is not None else None
        if text:
            parts.append(LMThinkingPart(text=text))
    return parts


def _message_parts(item: dict[str, Any]) -> list[LMPart]:
    """Convert an assistant message item's text and refusal content."""
    parts: list[LMPart] = []
    for raw_content in get_list(item, "content") or []:
        content = as_json_dict(raw_content)
        if content is None:
            continue
        content_type = get_str(content, "type")
        if content_type == "output_text":
            parts.append(LMTextPart(text=get_str(content, "text") or ""))
        elif content_type == "refusal":
            parts.append(LMRefusalPart(text=get_str(content, "refusal") or ""))
    return parts


def _tool_call_part(item: dict[str, Any]) -> LMToolCallPart:
    """Convert a function call item, keeping unparseable arguments as raw data."""
    arguments = get_str(item, "arguments")
    args: dict[str, Any] = {}
    if arguments:
        try:
            args = as_json_dict(json.loads(arguments)) or {}
        except json.JSONDecodeError:
            args = {}
    return LMToolCallPart(
        id=get_str(item, "call_id"),
        name=get_str(item, "name") or "",
        args=args,
        provider_data={"arguments": arguments} if arguments and not args else {},
    )


def _usage(usage: dict[str, Any]) -> LMUsage:
    """Map Responses usage counters to DSPy usage."""
    input_details = get_dict(usage, "input_tokens_details")
    output_details = get_dict(usage, "output_tokens_details")
    return LMUsage(
        input_tokens=get_int(usage, "input_tokens"),
        output_tokens=get_int(usage, "output_tokens"),
        total_tokens=get_int(usage, "total_tokens"),
        reasoning_tokens=(
            get_int(output_details, "reasoning_tokens") if output_details is not None else None
        ),
        cache_read_tokens=(
            get_int(input_details, "cached_tokens") if input_details is not None else None
        ),
        cache_write_tokens=(
            get_int(input_details, "cache_write_tokens") if input_details is not None else None
        ),
    )
