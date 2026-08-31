"""Error normalization and contract enforcement at the provider boundary."""

from __future__ import annotations

import dspy
import pytest

from conftest import CountingProvider, RecordingCallback
from dspy_base_lm import CustomLM, LMProvider


class NativeRateLimitError(Exception):
    """Representative error raised by a provider SDK."""


class KnownErrorProvider(LMProvider):
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        message = "slow down"
        raise dspy.LMRateLimitError(message, model=request.model, provider="known")

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class UnknownErrorProvider(LMProvider):
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = request, num_retries
        message = "native SDK failure"
        raise RuntimeError(message)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class InvalidResultProvider(LMProvider):
    """Deliberately violate the typed result contract to exercise rejection."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = request, num_retries
        self.calls += 1
        return iter([dspy.LMResponse.from_text("partial")])

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class MappingProvider(LMProvider):
    """Own a native retry loop and map the final failure to a DSPy error."""

    def __init__(self) -> None:
        super().__init__()
        self.native_attempts = 0

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        for attempt in range(num_retries + 1):
            self.native_attempts += 1
            try:
                return self._native_complete()
            except NativeRateLimitError as error:
                if attempt < num_retries:
                    continue
                message = "provider retry budget exhausted"
                raise dspy.LMRateLimitError(
                    message,
                    model=request.model,
                    provider="mapping",
                    status=429,
                    request_id="request-123",
                    retry_after=1.5,
                ) from error
        message = "the retry loop must either return or raise"
        raise AssertionError(message)

    @staticmethod
    def _native_complete() -> dspy.LMResponse:
        message = "native request was rate limited"
        raise NativeRateLimitError(message)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class FailingProvider(LMProvider):
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = request, num_retries
        message = "native failure"
        raise RuntimeError(message)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


def test_known_dspy_errors_pass_through_unchanged() -> None:
    # Given a provider that maps its backend failure to a DSPy error
    lm = CustomLM(model="errors/known", provider=KnownErrorProvider(), cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="fail")

    # When the call fails
    with pytest.raises(dspy.LMRateLimitError) as caught:
        lm.forward(request)

    # Then CustomLM preserves the provider's structured error
    assert caught.value.provider == "known"
    assert caught.value.__cause__ is None


def test_unknown_provider_errors_are_chained_at_the_boundary() -> None:
    # Given a provider that leaks an unknown native exception
    lm = CustomLM(model="errors/unknown", provider=UnknownErrorProvider(), cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="fail")

    # When the call crosses the CustomLM provider boundary
    with pytest.raises(dspy.LMUnexpectedError) as caught:
        lm.forward(request)

    # Then callers receive a DSPy error without losing the native cause
    assert caught.value.provider == "UnknownErrorProvider"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_invalid_provider_results_are_rejected_and_never_cached() -> None:
    # Given a provider that violates the completed-response contract
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    provider = InvalidResultProvider()
    lm = CustomLM(model="policy/invalid-result", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="reject partial output")

    # When the same invalid result is attempted twice through the public boundary
    for _ in range(2):
        with pytest.raises(TypeError, match="LMResponse"):
            _ = lm(request)

    # Then DSPy's own contract validation rejects it and nothing enters the cache
    assert provider.calls == 2


def test_provider_owns_retries_and_error_metadata() -> None:
    # Given a provider with a native retry and error-mapping policy
    provider = MappingProvider()
    lm = CustomLM(model="policy/errors", provider=provider, num_retries=2, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="map this failure")

    # When the provider exhausts its native retry budget
    with pytest.raises(dspy.LMRateLimitError) as caught:
        lm.forward(request)

    # Then CustomLM preserves the provider's DSPy error and native cause
    assert provider.native_attempts == 3
    assert caught.value.provider == "mapping"
    assert caught.value.status == 429
    assert caught.value.request_id == "request-123"
    assert caught.value.retry_after == 1.5
    assert isinstance(caught.value.__cause__, NativeRateLimitError)


@pytest.mark.parametrize(
    "location",
    ["extensions", "metadata", "message_metadata", "response_format", "cyclic_metadata"],
)
def test_request_runtime_objects_are_rejected_before_provider_or_cache(
    location: str,
    memory_cache: None,
    counting_provider: CountingProvider,
) -> None:
    # Given a runtime object or cycle hidden behind an innocuous request key
    provider = counting_provider
    lm = CustomLM(model="safe/request-runtime", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same")
    if location == "extensions":
        request.config.extensions["backend"] = object()
    elif location == "metadata":
        request.metadata["backend"] = object()
    elif location == "message_metadata":
        request.messages[0].metadata["backend"] = object()
    elif location == "response_format":
        request.config.response_format = object()
    else:
        request.metadata["nested"] = request.metadata

    # When the request crosses the LM boundary
    with pytest.raises(dspy.LMConfigurationError, match="JSON-like"):
        lm.forward(request)

    # Then validation runs before provider dispatch or cache identity creation
    assert provider.sync_calls == 0


def test_callback_observes_the_final_normalized_failure_once(
    recording_callback: RecordingCallback,
) -> None:
    # Given a provider failure and a standard DSPy callback
    callback = recording_callback
    lm = CustomLM(
        model="runtime/callback-failure",
        provider=FailingProvider(),
        callbacks=[callback],
        cache=False,
    )
    request = dspy.LMRequest.from_call(model=lm.model, prompt="fail once")

    # When the public DSPy call crosses the provider boundary
    with pytest.raises(dspy.LMUnexpectedError) as caught:
        lm(request)

    # Then inherited callback handling receives the same final failure exactly once
    assert callback.starts == callback.ends == 1
    assert callback.last_exception is caught.value
    assert lm.history == []
