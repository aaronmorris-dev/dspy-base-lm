"""Translate normalized DSPy requests into Responses API request bodies."""

from __future__ import annotations

import json
from itertools import count
from typing import TYPE_CHECKING, Any

import dspy
from dspy.core import (
    LMImagePart,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMToolResultPart,
)
from pydantic import BaseModel

from .errors import PROVIDER_NAME
from .json_values import as_json_dict, get_dict, get_str

if TYPE_CHECKING:
    from dspy.core import LMMessage, LMReasoningConfig, LMToolChoice, LMToolSpec

MODEL_PREFIX = "codex/"
_DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


def build_request_body(request: dspy.LMRequest) -> dict[str, Any]:
    """Build the Responses API request body for one normalized request."""
    _reject_unsupported_config(request)
    instructions, input_items = _instructions_and_items(request)
    body: dict[str, Any] = {
        "model": _native_model(request),
        "store": False,  # The Codex backend rejects stored responses.
        "stream": True,
        "instructions": instructions,
        "input": input_items,
    }
    config = request.config
    if config.reasoning is not None:
        reasoning_options = _reasoning_options(config.reasoning)
        if reasoning_options:
            body["reasoning"] = reasoning_options
    if config.response_format is not None:
        body["text"] = {"format": _response_format(config.response_format, request.model)}
    prompt_cache = config.prompt_cache
    if (
        prompt_cache is not None
        and prompt_cache.key is not None
        and prompt_cache.enabled is not False
    ):
        body["prompt_cache_key"] = prompt_cache.key
    if request.tools:
        body["tools"] = [_tool_definition(spec) for spec in request.tools]
    if config.tool_choice is not None:
        body["tool_choice"] = _tool_choice(config.tool_choice, request.model)
        if config.tool_choice.parallel is not None:
            body["parallel_tool_calls"] = config.tool_choice.parallel
    return body


def _native_model(request: dspy.LMRequest) -> str:
    """Strip the ``codex/`` prefix that names this provider's models."""
    native_model = request.model.removeprefix(MODEL_PREFIX)
    if native_model == request.model or not native_model:
        message = (
            f"CodexProvider expects models named '{MODEL_PREFIX}<codex-model>', "
            f"got {request.model!r}."
        )
        raise dspy.LMUnsupportedModelError(message, model=request.model, provider=PROVIDER_NAME)
    return native_model


def _reject_unsupported_config(request: dspy.LMRequest) -> None:
    """Reject config fields the Responses endpoint cannot honor, never drop them."""
    config = request.config
    reasoning_max_tokens = None
    if config.reasoning is not None:
        reasoning_max_tokens = config.reasoning.max_tokens
    unsupported = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stop": config.stop,
        "logprobs": config.logprobs,
        "reasoning.max_tokens": reasoning_max_tokens,
    }
    rejected = [name for name, value in unsupported.items() if value is not None]
    if config.n is not None and config.n != 1:
        rejected.append("n")
    prompt_cache = config.prompt_cache
    if prompt_cache is not None and prompt_cache.enabled and prompt_cache.key is None:
        rejected.append("prompt_cache.enabled")
    if config.extensions:
        rejected.extend(f"extensions.{key}" for key in config.extensions)
    if rejected:
        message = f"The Codex backend cannot honor request config: {', '.join(sorted(rejected))}."
        raise dspy.LMUnsupportedFeatureError(
            message,
            features=sorted(rejected),
            model=request.model,
            provider=PROVIDER_NAME,
        )


def _instructions_and_items(request: dspy.LMRequest) -> tuple[str, list[dict[str, Any]]]:
    """Split leading system messages into ``instructions`` and convert the rest."""
    system_texts: list[str] = []
    first_input_index = 0
    for message in request.messages:
        if message.role != "system":
            break
        system_texts.append(_text_only(message, request.model))
        first_input_index += 1

    instructions = "\n\n".join(system_texts) or _DEFAULT_INSTRUCTIONS
    input_messages = request.messages[first_input_index:]
    return instructions, _input_items(input_messages, request.model)


def _input_items(messages: list[LMMessage], model: str) -> list[dict[str, Any]]:
    """Convert conversation messages into Responses input items.

    Tool calls and tool results become their own top-level items. When DSPy
    replays a conversation without provider call ids, matching positional ids
    are synthesized so each result still pairs with its call.
    """
    items: list[dict[str, Any]] = []
    synthesized_call_ids = count()
    synthesized_result_ids = count()
    for message in messages:
        role = message.role
        if role == "system":
            role = "developer"
        message_content: list[dict[str, Any]] = []
        separate_items: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, LMTextPart):
                message_content.append(_text_content(role, part.text))
            elif isinstance(part, LMImagePart):
                message_content.append(_image_content(part, model))
            elif isinstance(part, LMToolCallPart):
                call_id = part.id or f"call_{next(synthesized_call_ids)}"
                separate_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": part.name,
                        "arguments": json.dumps(part.args),
                    }
                )
            elif isinstance(part, LMToolResultPart):
                call_id = part.call_id or f"call_{next(synthesized_result_ids)}"
                separate_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": _tool_result_text(part, model),
                    }
                )
            elif isinstance(part, LMThinkingPart):
                continue  # Reasoning is not replayable without encrypted content.
            else:
                raise _unsupported_part(part.type, model)
        if message_content:
            items.append({"type": "message", "role": role, "content": message_content})
        items.extend(separate_items)
    return items


def _text_content(role: str, text: str) -> dict[str, str]:
    """Wrap text as input or output content depending on who said it."""
    content_type = "input_text"
    if role == "assistant":
        content_type = "output_text"
    return {"type": content_type, "text": text}


def _image_content(part: LMImagePart, model: str) -> dict[str, Any]:
    """Convert an image part sourced from a URL or inline base64 data."""
    if part.url is not None:
        image_url = part.url
    elif part.data is not None:
        image_url = f"data:{part.media_type};base64,{part.data}"
    else:
        raise _unsupported_part("image_file", model)
    content: dict[str, Any] = {"type": "input_image", "image_url": image_url}
    if part.detail is not None:
        content["detail"] = part.detail
    return content


def _text_only(message: LMMessage, model: str) -> str:
    """Join a message's text parts, rejecting any other content."""
    for part in message.parts:
        if not isinstance(part, LMTextPart):
            raise _unsupported_part(f"system_{part.type}", model)
    return message.text or ""


def _tool_result_text(part: LMToolResultPart, model: str) -> str:
    """Join a tool result's text content, rejecting any other content."""
    texts: list[str] = []
    for item in part.content:
        if not isinstance(item, LMTextPart):
            raise _unsupported_part(f"tool_result_{item.type}", model)
        texts.append(item.text)
    return "\n".join(texts)


def _unsupported_part(feature: str, model: str) -> dspy.LMUnsupportedFeatureError:
    message = f"The Codex backend cannot honor {feature} content."
    return dspy.LMUnsupportedFeatureError(
        message,
        features=[feature],
        model=model,
        provider=PROVIDER_NAME,
    )


def _reasoning_options(reasoning_config: LMReasoningConfig) -> dict[str, Any]:
    """Convert reasoning controls; ``max_tokens`` was already rejected."""
    reasoning: dict[str, Any] = {}
    if reasoning_config.effort is not None:
        reasoning["effort"] = reasoning_config.effort
    if reasoning_config.summary is not None:
        reasoning["summary"] = reasoning_config.summary
    return reasoning


def _tool_definition(spec: LMToolSpec) -> dict[str, Any]:
    """Convert one provider-independent tool spec to a Responses function tool."""
    tool: dict[str, Any] = {
        "type": "function",
        "name": spec.name,
        "parameters": spec.parameters,
        "strict": bool(spec.strict),
    }
    if spec.description is not None:
        tool["description"] = spec.description
    return tool


def _tool_choice(choice: LMToolChoice, model: str) -> str | dict[str, Any]:
    """Convert DSPy's tool-choice controls to the Responses equivalent."""
    if not choice.allowed:
        return choice.mode
    if len(choice.allowed) == 1:
        return {"type": "function", "name": choice.allowed[0]}
    message = "The Codex backend cannot restrict tool choice to a set of tools."
    raise dspy.LMUnsupportedFeatureError(
        message,
        features=["tool_choice.allowed"],
        model=model,
        provider=PROVIDER_NAME,
    )


def _response_format(response_format: object, model: str) -> dict[str, Any]:
    """Convert DSPy's ``response_format`` value to a Responses text format.

    Accepts a Pydantic model class (DSPy's JSONAdapter passes one when the
    provider reports native schema support), an OpenAI chat-style
    ``json_schema`` wrapper, a Responses-style format object, a bare JSON
    schema, or ``{"type": "json_object"}``.
    """
    if isinstance(response_format, type):
        if issubclass(response_format, BaseModel):
            return {
                "type": "json_schema",
                "name": response_format.__name__,
                "strict": False,
                "schema": response_format.model_json_schema(),
            }
        raise _unsupported_response_format(response_format.__name__, model)
    format_data = as_json_dict(response_format)
    if format_data is None:
        raise _unsupported_response_format(type(response_format).__name__, model)
    format_type = get_str(format_data, "type")
    if format_type == "json_object":
        return {"type": "json_object"}
    if format_type == "json_schema":
        wrapped_schema = get_dict(format_data, "json_schema")
        if wrapped_schema is None:
            return format_data
        return {
            "type": "json_schema",
            "name": get_str(wrapped_schema, "name") or "response",
            "strict": bool(wrapped_schema.get("strict")),
            "schema": get_dict(wrapped_schema, "schema") or {},
        }
    if format_type is None:
        return {
            "type": "json_schema",
            "name": "response",
            "strict": False,
            "schema": format_data,
        }
    raise _unsupported_response_format(format_type, model)


def _unsupported_response_format(description: str, model: str) -> dspy.LMUnsupportedFeatureError:
    message = f"The Codex backend cannot honor response_format {description!r}."
    return dspy.LMUnsupportedFeatureError(
        message,
        features=["response_format"],
        model=model,
        provider=PROVIDER_NAME,
    )
