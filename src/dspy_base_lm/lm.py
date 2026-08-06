"""A thin typed language model that delegates transport work to a provider."""

import threading
from typing import Any, Literal, TypeAlias

import dspy
from dspy.utils.callback import BaseCallback
from typing_extensions import override

from dspy_base_lm._cache import (
    UncacheableResponseError,
    find_runtime_config_path,
    provider_cache_partition,
    response_is_cache_safe,
)
from dspy_base_lm._dspy import ForwardContract, TrainDataFormat, request_cache
from dspy_base_lm._errors import ProviderErrorBoundary, TrainingJobErrorBoundary
from dspy_base_lm.provider import LMProvider

_ConfigValue: TypeAlias = (
    bool | int | float | str | list["_ConfigValue"] | dict[str, "_ConfigValue"] | None
)


class CustomLM(dspy.BaseLM):
    """A provider-backed base class for typed custom DSPy language models."""

    forward_contract: ForwardContract = "typed_lm"

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
        **kwargs: _ConfigValue,
    ) -> None:
        """Create a typed custom LM around an injected or inferred provider."""
        persistent_config = dict(kwargs)
        if extensions is not None:
            persistent_config["extensions"] = dict(extensions)
        normalized_config = dspy.LMConfig.from_kwargs(
            temperature=temperature,
            max_tokens=max_tokens,
            **persistent_config,
        )
        self._reject_runtime_config(normalized_config.extensions, prefix="extensions")
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **persistent_config,
        )
        self.model: str = model
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

    def finetune(
        self,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> dspy.TrainingJob:
        """Start DSPy's provider-owned fine-tuning job orchestration."""
        if not self.provider.finetunable:
            message = f"{type(self.provider).__name__} does not support fine-tuning."
            raise dspy.LMUnsupportedFeatureError(
                message,
                model=self.model,
                provider=type(self.provider).__name__,
                features=["finetuning"],
            )

        job: dspy.TrainingJob

        def run_job() -> None:
            with TrainingJobErrorBoundary(job):
                model = self.provider.finetune(
                    job=job,
                    model=self.model,
                    train_data=train_data,
                    train_data_format=train_data_format,
                    train_kwargs=train_kwargs,
                )
                job.set_result(self.copy(model=model))

        thread = threading.Thread(target=run_job)
        job = self.provider.TrainingJob(
            thread=thread,
            model=self.model,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs,
        )
        thread.start()
        return job

    @override
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        """Dispatch one normalized request through the synchronous provider boundary."""
        self._reject_request_runtime_state(request)
        if self._cache_enabled(request):
            try:
                return self._cached_complete(
                    cache_request=self._cache_identity(request),
                    request=request,
                )
            except UncacheableResponseError as error:
                return error.response
        return self._complete(request)

    @override
    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        """Dispatch one normalized request through the asynchronous provider boundary."""
        self._reject_request_runtime_state(request)
        if self._cache_enabled(request):
            try:
                return await self._cached_acomplete(
                    cache_request=self._cache_identity(request),
                    request=request,
                )
            except UncacheableResponseError as error:
                return error.response
        return await self._acomplete(request)

    @request_cache(cache_arg_name="cache_request")
    def _cached_complete(
        self,
        *,
        cache_request: dict[str, Any],
        request: dspy.LMRequest,
    ) -> dspy.LMResponse:
        _ = cache_request
        response = self._complete(request)
        if not response_is_cache_safe(response):
            raise UncacheableResponseError(response)
        return response

    @request_cache(cache_arg_name="cache_request")
    async def _cached_acomplete(
        self,
        *,
        cache_request: dict[str, Any],
        request: dspy.LMRequest,
    ) -> dspy.LMResponse:
        _ = cache_request
        response = await self._acomplete(request)
        if not response_is_cache_safe(response):
            raise UncacheableResponseError(response)
        return response

    def _complete(self, request: dspy.LMRequest) -> dspy.LMResponse:
        with ProviderErrorBoundary(self._unexpected_error):
            response = self.provider.complete(request, num_retries=self.num_retries)
        return self._validate_typed_lm_response(response)

    async def _acomplete(self, request: dspy.LMRequest) -> dspy.LMResponse:
        with ProviderErrorBoundary(self._unexpected_error):
            response = await self.provider.acomplete(request, num_retries=self.num_retries)
        return self._validate_typed_lm_response(response)

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

    def _cache_identity(self, request: dspy.LMRequest) -> dict[str, Any]:
        cache_config = request.config.cache
        if cache_config is None or cache_config.rollout_id is None:
            normalized_cache = None
        else:
            normalized_cache = cache_config.model_copy(update={"enabled": None})
        config = request.config.model_copy(
            update={"cache": normalized_cache},
        )
        normalized_request = request.model_copy(
            update={"config": config, "metadata": {}},
            deep=True,
        )
        identity = normalized_request.model_dump(mode="python")
        provider_type = type(self.provider)
        identity["provider_type"] = f"{provider_type.__module__}.{provider_type.__qualname__}"
        identity["provider_partition"] = provider_cache_partition(self.provider)
        return identity

    @classmethod
    def _reject_runtime_config(
        cls,
        kwargs: dict[str, _ConfigValue],
        *,
        prefix: str = "",
    ) -> None:
        runtime_path = find_runtime_config_path(kwargs, prefix=prefix.rstrip("."))
        if runtime_path is not None:
            cls._raise_runtime_config_error(runtime_path)

    @classmethod
    def _reject_request_runtime_state(cls, request: dspy.LMRequest) -> None:
        extension_path = find_runtime_config_path(
            request.config.extensions,
            prefix="request.config.extensions",
        )
        if extension_path is not None:
            cls._raise_runtime_config_error(extension_path)
        metadata_path = find_runtime_config_path(request.metadata, prefix="request.metadata")
        if metadata_path is not None:
            cls._raise_runtime_config_error(metadata_path)

    @staticmethod
    def _raise_runtime_config_error(key: str) -> None:
        message = (
            f"{key!r} is not safe persistent LM configuration. Extension and metadata "
            "values must be JSON-like, while clients and credentials belong on LMProvider."
        )
        raise dspy.LMConfigurationError(message)
