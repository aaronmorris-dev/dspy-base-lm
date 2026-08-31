"""Run DSPy programs on a ChatGPT/Codex subscription through the Codex CLI.

`CodexProvider` owns the subprocess transport: `codex exec --json` argv
construction, JSONL stdout framing, backend error mapping, timeouts, and
retries. `CodexLM` is the reconstructable `CustomLM` subclass: authentication
is ambient (`codex login`), so no state beyond DSPy LM state is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

import dspy
from dspy.core import LMOutput, LMTextPart, LMThinkingPart, LMUsage
from typing_extensions import override

from dspy_base_lm import CustomLM, LMProvider

if TYPE_CHECKING:
    from collections.abc import Generator

    from dspy.core import LMPart

_MODEL_PREFIX = "codex/"
_PROVIDER_NAME = "codex"
_RETRYABLE_ERRORS = (dspy.LMRateLimitError, dspy.LMServerError)
_MAX_BACKOFF_SECONDS = 30.0
_STDERR_TAIL_CHARS = 500


class CodexProvider(LMProvider):
    """Translate DSPy requests into non-interactive ``codex exec`` runs.

    Each request becomes one ephemeral, read-only-sandboxed Codex turn whose
    final agent message is the completion text. The Codex CLI holds the
    ChatGPT-subscription credentials, so this provider carries no secrets.
    """

    def __init__(self, *, codex_binary: str = "codex", timeout: float = 600.0) -> None:
        """Configure the Codex CLI binary to spawn and a per-call deadline."""
        super().__init__()
        self.codex_binary: str = codex_binary
        self.timeout: float = timeout

    @override
    def supports_reasoning(self, model: str) -> bool:
        """Codex models reason natively; effort maps to ``model_reasoning_effort``."""
        _ = model
        return True

    @override
    def supported_params(self, model: str) -> frozenset[str]:
        """Return the request config fields ``codex exec`` can honor."""
        _ = model
        return frozenset({"reasoning", "response_format"})

    @override
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        """Complete one request synchronously, retrying transient failures."""
        argv, prompt, schema = self._translate(request)
        for attempt in range(num_retries):
            try:
                return self._run_sync(argv, prompt, schema, model=request.model)
            except _RETRYABLE_ERRORS:
                time.sleep(_backoff_seconds(attempt))
        return self._run_sync(argv, prompt, schema, model=request.model)

    @override
    async def acomplete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        """Complete one request asynchronously, retrying transient failures."""
        argv, prompt, schema = self._translate(request)
        for attempt in range(num_retries):
            try:
                return await self._run_async(argv, prompt, schema, model=request.model)
            except _RETRYABLE_ERRORS:
                await asyncio.sleep(_backoff_seconds(attempt))
        return await self._run_async(argv, prompt, schema, model=request.model)

    def _translate(
        self,
        request: dspy.LMRequest,
    ) -> tuple[list[str], str, dict[str, Any] | None]:
        """Map one normalized request to argv, a prompt, and an optional schema."""
        native_model = request.model.removeprefix(_MODEL_PREFIX)
        if native_model == request.model or not native_model:
            message = (
                f"CodexProvider expects models named '{_MODEL_PREFIX}<codex-model>', "
                f"got {request.model!r}."
            )
            raise dspy.LMUnsupportedModelError(
                message,
                model=request.model,
                provider=_PROVIDER_NAME,
            )
        if request.tools:
            message = "codex exec does not accept caller-defined tools."
            raise dspy.LMUnsupportedFeatureError(
                message,
                features=["function_calling"],
                model=request.model,
                provider=_PROVIDER_NAME,
            )
        _reject_unsupported_config(request)
        argv = [
            self.codex_binary,
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--model",
            native_model,
        ]
        reasoning = request.config.reasoning
        if reasoning is not None and reasoning.effort is not None:
            argv.extend(["-c", f'model_reasoning_effort="{reasoning.effort}"'])
        argv.append("-")  # Read the prompt from stdin.
        return argv, _render_prompt(request), _response_schema(request)

    def _run_sync(
        self,
        argv: list[str],
        prompt: str,
        schema: dict[str, Any] | None,
        *,
        model: str,
    ) -> dspy.LMResponse:
        with _schema_file(argv, schema) as full_argv:
            try:
                completed = subprocess.run(
                    full_argv,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except FileNotFoundError as error:
                raise self._not_installed(model) from error
            except subprocess.TimeoutExpired as error:
                raise self._timeout(model) from error
        return self._to_response(
            completed.returncode,
            completed.stdout,
            completed.stderr,
            model=model,
        )

    async def _run_async(
        self,
        argv: list[str],
        prompt: str,
        schema: dict[str, Any] | None,
        *,
        model: str,
    ) -> dspy.LMResponse:
        with _schema_file(argv, schema) as full_argv:
            try:
                process = await asyncio.create_subprocess_exec(
                    *full_argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as error:
                raise self._not_installed(model) from error
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(prompt.encode()),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError as error:
                process.kill()
                _ = await process.wait()
                raise self._timeout(model) from error
        returncode = process.returncode if process.returncode is not None else -1
        return self._to_response(
            returncode,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            model=model,
        )

    def _to_response(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        *,
        model: str,
    ) -> dspy.LMResponse:
        """Map one finished ``codex exec`` run to a typed response or error."""
        events = _parse_events(stdout)
        failure = next(
            (event for event in events if event.get("type") in {"error", "turn.failed"}),
            None,
        )
        if failure is not None:
            raise _backend_error(failure, model=model)
        if returncode != 0:
            message = (
                f"codex exec exited with status {returncode}: "
                f"{stderr.strip()[-_STDERR_TAIL_CHARS:]}"
            )
            raise dspy.LMTransportError(message, model=model, provider=_PROVIDER_NAME)

        turn = _collect_turn(events)
        if not turn.texts:
            message = "codex exec finished without an agent message."
            raise dspy.LMTransportError(message, model=model, provider=_PROVIDER_NAME)

        parts: list[LMPart] = [LMThinkingPart(text=text) for text in turn.thinking]
        parts.append(LMTextPart(text="\n\n".join(turn.texts)))
        return dspy.LMResponse(
            model=model,
            outputs=[LMOutput(parts=parts, finish_reason="stop")],
            usage=turn.usage,
            response_id=turn.thread_id,
            provider_response={"events": events},
        )

    def _not_installed(self, model: str) -> dspy.LMNotConfiguredError:
        message = (
            f"Codex CLI binary {self.codex_binary!r} was not found. Install the Codex "
            "CLI and sign in with `codex login` (ChatGPT subscription)."
        )
        return dspy.LMNotConfiguredError(message, model=model, provider=_PROVIDER_NAME)

    def _timeout(self, model: str) -> dspy.LMTimeoutError:
        message = f"codex exec exceeded the {self.timeout:g}s provider deadline."
        return dspy.LMTimeoutError(message, model=model, provider=_PROVIDER_NAME)


class CodexLM(CustomLM):
    """A reconstructable Codex-subscription LM.

    Credentials live in the Codex CLI (``codex login``), so the provider is
    inferable from ambient configuration and ``dump_state()`` stays safe.
    """

    @override
    def infer_provider(self) -> LMProvider:
        """Reconstruct the provider from the ambient Codex CLI installation."""
        return CodexProvider()


@dataclass
class _Turn:
    """The pieces of one completed ``codex exec`` turn."""

    texts: list[str] = field(default_factory=list)
    thinking: list[str] = field(default_factory=list)
    usage: LMUsage | None = None
    thread_id: str | None = None


def _collect_turn(events: list[dict[str, Any]]) -> _Turn:
    """Gather agent text, reasoning, usage, and the thread ID from turn events."""
    turn = _Turn()
    for event in events:
        event_type = _get_str(event, "type")
        if event_type == "thread.started":
            turn.thread_id = _get_str(event, "thread_id") or turn.thread_id
        elif event_type == "item.completed":
            item = _get_dict(event, "item") or {}
            text = _get_str(item, "text")
            if text is None:
                continue
            if _get_str(item, "type") == "agent_message":
                turn.texts.append(text)
            elif _get_str(item, "type") == "reasoning":
                turn.thinking.append(text)
        elif event_type == "turn.completed":
            usage = _get_dict(event, "usage")
            if usage is not None:
                turn.usage = _to_usage(usage)
    return turn


def _backoff_seconds(attempt: int) -> float:
    return min(2.0**attempt, _MAX_BACKOFF_SECONDS)


def _reject_unsupported_config(request: dspy.LMRequest) -> None:
    """Reject config fields ``codex exec`` cannot honor rather than drop them."""
    config = request.config
    unsupported = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": config.top_p,
        "stop": config.stop,
        "logprobs": config.logprobs,
        "tool_choice": config.tool_choice,
        "prompt_cache": config.prompt_cache,
        "reasoning.max_tokens": config.reasoning and config.reasoning.max_tokens,
        "reasoning.summary": config.reasoning and config.reasoning.summary,
    }
    rejected = [name for name, value in unsupported.items() if value is not None]
    if config.n is not None and config.n != 1:
        rejected.append("n")
    if config.extensions:
        rejected.extend(f"extensions.{key}" for key in config.extensions)
    if rejected:
        message = f"codex exec cannot honor request config: {', '.join(sorted(rejected))}."
        raise dspy.LMUnsupportedFeatureError(
            message,
            features=sorted(rejected),
            model=request.model,
            provider=_PROVIDER_NAME,
        )


def _render_prompt(request: dspy.LMRequest) -> str:
    """Flatten the text-only message transcript into one ``codex exec`` prompt."""
    sections: list[str] = []
    for message in request.messages:
        for part in message.parts:
            if not isinstance(part, LMTextPart):
                message_text = f"codex exec accepts text messages only, got a {part.type} part."
                raise dspy.LMUnsupportedFeatureError(
                    message_text,
                    features=[f"{part.type}_input"],
                    model=request.model,
                    provider=_PROVIDER_NAME,
                )
        sections.append(f"[{message.role}]\n{message.text or ''}")
    return "\n\n".join(sections)


def _response_schema(request: dspy.LMRequest) -> dict[str, Any] | None:
    """Extract the JSON schema, if any, to pass as ``--output-schema``.

    ``{"type": "json_object"}`` needs no schema file: the DSPy adapter prompt
    already demands a JSON object. OpenAI-style ``json_schema`` wrappers are
    unwrapped; any other dict is treated as a JSON schema itself.
    """
    response_format: object = request.config.response_format
    if response_format is None:
        return None
    formats = _as_json_dict(response_format)
    if formats is None:
        message = (
            f"codex exec cannot honor response_format of type {type(response_format).__name__}."
        )
        raise dspy.LMUnsupportedFeatureError(
            message,
            features=["response_format"],
            model=request.model,
            provider=_PROVIDER_NAME,
        )
    if _get_str(formats, "type") == "json_object":
        return None
    wrapped = _get_dict(formats, "json_schema")
    if wrapped is not None:
        schema = _get_dict(wrapped, "schema")
        if schema is not None:
            return schema
    return formats


@contextlib.contextmanager
def _schema_file(
    argv: list[str],
    schema: dict[str, Any] | None,
) -> Generator[list[str], None, None]:
    """Yield argv, extended with a temporary ``--output-schema`` file if needed."""
    if schema is None:
        yield argv
        return
    with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="dspy-codex-schema-") as file:
        json.dump(schema, file)
        file.flush()
        yield [*argv[:-1], "--output-schema", file.name, argv[-1]]


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    """Parse JSONL events from stdout, skipping non-JSON warning lines."""
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event: object = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        parsed = _as_json_dict(event)
        if parsed is not None:
            events.append(parsed)
    return events


def _to_usage(usage: dict[str, Any]) -> LMUsage:
    return LMUsage(
        input_tokens=_get_int(usage, "input_tokens"),
        output_tokens=_get_int(usage, "output_tokens"),
        reasoning_tokens=_get_int(usage, "reasoning_output_tokens"),
        cache_read_tokens=_get_int(usage, "cached_input_tokens"),
        cache_write_tokens=_get_int(usage, "cache_write_input_tokens"),
    )


def _backend_error(failure: dict[str, Any], *, model: str) -> dspy.LMError:
    """Map one ``error`` or ``turn.failed`` event to the most specific DSPy error."""
    error = _get_dict(failure, "error")
    raw = _get_str(error, "message") if error is not None else _get_str(failure, "message")
    raw_message = raw if raw is not None else "codex exec reported an unknown error."
    message, status = _unwrap_backend_message(raw_message)
    error_type = _classify_backend_error(status, message.lower())
    return error_type(message=message, model=model, provider=_PROVIDER_NAME, status=status)


def _classify_backend_error(status: int | None, lowered: str) -> type[dspy.LMError]:
    """Choose the most specific DSPy error type for a backend failure."""
    error_type: type[dspy.LMError]
    if "context window" in lowered:
        error_type = dspy.ContextWindowExceededError
    elif (
        status in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}
        or "not logged in" in lowered
        or "unauthorized" in lowered
    ):
        error_type = dspy.LMAuthError
    elif status == HTTPStatus.PAYMENT_REQUIRED or "billing" in lowered:
        error_type = dspy.LMBillingError
    elif (
        status == HTTPStatus.TOO_MANY_REQUESTS
        or "rate limit" in lowered
        or "usage limit" in lowered
    ):
        error_type = dspy.LMRateLimitError
    elif status is not None and status >= HTTPStatus.INTERNAL_SERVER_ERROR:
        error_type = dspy.LMServerError
    elif "model is not supported" in lowered:
        error_type = dspy.LMUnsupportedModelError
    elif status is not None:
        error_type = dspy.LMInvalidRequestError
    else:
        error_type = dspy.LMProviderError
    return error_type


def _unwrap_backend_message(raw_message: str) -> tuple[str, int | None]:
    """Unwrap the JSON error body Codex embeds in failure messages, if present."""
    try:
        parsed: object = json.loads(raw_message)
    except json.JSONDecodeError:
        return raw_message, None
    detail = _as_json_dict(parsed)
    if detail is None:
        return raw_message, None
    inner = _as_json_dict(detail.get("error"))
    inner_message = _get_str(inner, "message") if inner is not None else None
    return inner_message or raw_message, _get_int(detail, "status")


def _as_json_dict(value: object) -> dict[str, Any] | None:
    """Narrow parsed JSON to a string-keyed dict; JSON object keys are strings."""
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return None


def _get_dict(mapping: dict[str, Any], key: str) -> dict[str, Any] | None:
    return _as_json_dict(mapping.get(key))


def _get_str(mapping: dict[str, Any], key: str) -> str | None:
    value: object = mapping.get(key)
    return value if isinstance(value, str) else None


def _get_int(mapping: dict[str, Any], key: str) -> int | None:
    value: object = mapping.get(key)
    return value if isinstance(value, int) else None
