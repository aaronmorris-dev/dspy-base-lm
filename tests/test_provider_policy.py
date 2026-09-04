from __future__ import annotations

import dspy
import pytest

from dspy_base_lm import CustomLM, LMProvider


class NativeRateLimitError(Exception):
    """Representative error raised by a provider SDK."""


class MappingProvider(LMProvider):
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


class IteratorProvider(LMProvider):
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


def test_provider_owns_retries_and_maps_native_errors() -> None:
    provider = MappingProvider()
    lm = CustomLM(model="policy/errors", provider=provider, num_retries=2, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="map this failure")

    with pytest.raises(dspy.LMRateLimitError) as caught:
        lm.forward(request)

    assert provider.native_attempts == 3
    assert caught.value.provider == "mapping"
    assert caught.value.status == 429
    assert caught.value.request_id == "request-123"
    assert caught.value.retry_after == 1.5
    assert isinstance(caught.value.__cause__, NativeRateLimitError)


def test_iterator_results_are_rejected_and_never_cached() -> None:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    provider = IteratorProvider()
    lm = CustomLM(model="policy/iterator", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="reject partial output")

    for _ in range(2):
        with pytest.raises(TypeError, match="LMResponse"):
            lm.forward(request)

    assert provider.calls == 2
