"""A thin typed language model that delegates transport work to a provider."""

import json
from collections.abc import Callable
from types import TracebackType
from typing import Literal, TypeAlias, cast

import dspy
from dspy.core import LMCacheConfig, LMPromptCacheConfig, LMReasoningConfig, LMToolChoice
from dspy.utils.callback import BaseCallback
from pydantic import BaseModel
from typing_extensions import Self, override

from dspy_base_lm.provider import LMProvider

_ConfigValue: TypeAlias = (
    bool | int | float | str | list["_ConfigValue"] | dict[str, "_ConfigValue"] | None
)
_ConfigInput: TypeAlias = (
    _ConfigValue | LMCacheConfig | LMPromptCacheConfig | LMReasoningConfig | LMToolChoice
)


class CustomLM(dspy.BaseLM):
    """A provider-backed base class for typed custom DSPy language models."""

    _provider_injected: bool
    # Structurally identical to BaseLM's `ForwardContract` migration marker type.
    forward_contract: Literal["legacy", "typed_lm"] = "typed_lm"

    def __init__(  # noqa: PLR0913
        self,
        model: str,
        *,
        model_type: Literal["chat", "text", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: LMProvider | None = None,
        extensions: dict[str, _ConfigValue] | None = None,
        **kwargs: _ConfigInput,
    ) -> None:
        """Create a typed custom LM around an injected or inferred provider.

        Persistent configuration is stored flat, exactly as ``BaseLM`` stores
        ``self.kwargs``; DSPy groups it into typed request config per call.
        """
        persistent_config: dict[str, _ConfigInput] = dict(kwargs)
        if extensions is not None:
            persistent_config["extensions"] = dict(extensions)
        safe_config = {
            key: self._reconstruction_safe_value(key, value)
            for key, value in persistent_config.items()
        }
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **safe_config,
        )
        self.model: str = model
        self._provider_injected = provider is not None
        self.provider: LMProvider = provider if provider is not None else self.infer_provider()

    def infer_provider(self) -> LMProvider:
        """Construct a provider from persistent subclass configuration."""
        message = (
            "No provider is configured. Inject an LMProvider into CustomLM, or subclass "
            "CustomLM and implement infer_provider() for reconstructable configuration."
        )
        raise dspy.LMNotConfiguredError(
            message,
            model=self.model,
        )

    @property
    @override
    def supports_function_calling(self) -> bool:
        """Whether the attached provider supports native function calling."""
        return self.provider.supports_function_calling(self.model)

    @property
    @override
    def supports_reasoning(self) -> bool:
        """Whether the attached provider supports native reasoning."""
        return self.provider.supports_reasoning(self.model)

    @property
    @override
    def supports_response_schema(self) -> bool:
        """Whether the attached provider supports native response schemas."""
        return self.provider.supports_response_schema(self.model)

    @property
    @override
    def supported_params(self) -> set[str]:
        """Return provider-supported DSPy generation parameter names."""
        return set(self.provider.supported_params(self.model))

    @override
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        """Dispatch one normalized request through the synchronous provider boundary."""
        self._validate_request_state(request)
        if not self._cache_enabled(request):
            return self._complete(request)

        cache_request = self._cache_identity(request)
        cached = cast("object", dspy.cache.get(cache_request))
        if isinstance(cached, dspy.LMResponse):
            return cached

        response = self._complete(request)
        if _response_is_cacheable(response):
            dspy.cache.put(cache_request, response)
        return response

    @override
    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        """Dispatch one normalized request through the asynchronous provider boundary."""
        self._validate_request_state(request)
        if not self._cache_enabled(request):
            return await self._acomplete(request)

        cache_request = self._cache_identity(request)
        cached = cast("object", dspy.cache.get(cache_request))
        if isinstance(cached, dspy.LMResponse):
            return cached

        response = await self._acomplete(request)
        if _response_is_cacheable(response):
            dspy.cache.put(cache_request, response)
        return response

    def _complete(self, request: dspy.LMRequest) -> dspy.LMResponse:
        with _ProviderErrorBoundary(self._unexpected_error):
            return self.provider.complete(request, num_retries=self.num_retries)

    async def _acomplete(self, request: dspy.LMRequest) -> dspy.LMResponse:
        with _ProviderErrorBoundary(self._unexpected_error):
            return await self.provider.acomplete(request, num_retries=self.num_retries)

    def _unexpected_error(self, error: Exception) -> dspy.LMUnexpectedError:
        provider_name = type(self.provider).__name__
        return dspy.LMUnexpectedError(
            f"{provider_name} raised an unexpected {type(error).__name__}.",
            model=self.model,
            provider=provider_name,
        )

    def _cache_enabled(self, request: dspy.LMRequest) -> bool:
        cache_config = request.config.cache
        if cache_config is None or cache_config.enabled is None:
            return self.cache
        return cache_config.enabled

    def _cache_identity(self, request: dspy.LMRequest) -> dict[str, object]:
        cache_config = request.config.cache
        normalized_cache = (
            None
            if cache_config is None or cache_config.rollout_id is None
            else cache_config.model_copy(update={"enabled": None})
        )
        config_update: dict[str, object] = {"cache": normalized_cache}
        response_format = cast("object", request.config.response_format)
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            # Key Pydantic response formats by their schema, as dspy.cache does.
            config_update["response_format"] = response_format.model_json_schema()
        config = request.config.model_copy(update=config_update)
        normalized_request = request.model_copy(update={"config": config}, deep=True)
        return normalized_request.model_dump(mode="json")

    @classmethod
    def _reconstruction_safe_value(cls, key: str, value: _ConfigInput) -> object:
        """Validate and convert one persistent config value to finite JSON-like data.

        Native DSPy config objects are kept as their reconstruction-safe dumps
        so ``dump_state()`` output stays serializable.
        """
        safe_value: object = (
            value.model_dump(mode="python") if isinstance(value, BaseModel) else value
        )
        cls._reject_non_json_config(safe_value, prefix=key)
        return safe_value

    @classmethod
    def _reject_non_json_config(
        cls,
        config: object,
        *,
        prefix: str = "configuration",
    ) -> None:
        if not _is_finite_json(config):
            message = (
                f"{prefix!r} is not safe LM configuration. Persistent configuration "
                "and requests must contain only finite JSON-like data; runtime objects belong "
                "on LMProvider."
            )
            raise dspy.LMConfigurationError(message)

    @override
    def copy(self, **kwargs: object) -> Self:
        """Copy DSPy state while tracking runtime provider replacement."""
        copied = super().copy(**kwargs)
        if "provider" in kwargs:
            object.__setattr__(copied, "_provider_injected", True)
        return copied

    @override
    def dump_state(self) -> dict[str, object]:
        """Return reconstructable state without serializing an injected provider."""
        if self._provider_injected:
            message = (
                "CustomLM instances with an injected provider are runtime-only and cannot be "
                "reconstructed. Subclass CustomLM and implement infer_provider() before saving."
            )
            raise dspy.LMConfigurationError(message)
        return super().dump_state()

    @classmethod
    def _validate_request_state(cls, request: dspy.LMRequest) -> None:
        values = cast("dict[str, object]", request.model_dump(mode="python"))
        config = values.get("config")
        if isinstance(config, dict):
            values["config"] = _declarative_config_view(cast("dict[str, object]", config))
        cls._reject_non_json_config(values, prefix="request")


class _ProviderErrorBoundary:
    """Preserve DSPy errors and normalize unknown provider failures."""

    def __init__(self, normalize: Callable[[Exception], dspy.LMUnexpectedError]) -> None:
        self._normalize: Callable[[Exception], dspy.LMUnexpectedError] = normalize

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        _ = exception_type, traceback
        if exception is None or isinstance(exception, dspy.LMError):
            return False
        if isinstance(exception, Exception):
            raise self._normalize(exception) from exception
        return False


def _declarative_config_view(config: dict[str, object]) -> dict[str, object]:
    """Represent a Pydantic ``response_format`` class by its JSON schema.

    DSPy treats a Pydantic model class as valid declarative request config:
    ``dspy.LM`` forwards it to the transport, and ``dspy.cache`` keys it by
    ``model_json_schema()``. Mirror that so JSON-safety validation accepts it,
    while providers own the wire translation.
    """
    response_format = config.get("response_format")
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return {**config, "response_format": response_format.model_json_schema()}
    return config


def _response_is_cacheable(response: dspy.LMResponse) -> bool:
    """Return whether DSPy can safely persist the completed response."""
    try:
        values: object = response.model_dump(mode="python")
    except Exception:  # noqa: BLE001 - serialization failures simply bypass caching
        return False
    return (
        _is_finite_json(values)
        and _is_finite_json(response.provider_response)
        and all(_is_finite_json(output.provider_output) for output in response.outputs)
    )


def _is_finite_json(value: object) -> bool:
    try:
        _ = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True
