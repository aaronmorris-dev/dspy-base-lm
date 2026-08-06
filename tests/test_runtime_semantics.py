from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy
import pytest
from dspy.utils.callback import BaseCallback

from dspy_base_lm import CustomLM, LMProvider

if TYPE_CHECKING:
    from pathlib import Path


class UsageProvider(LMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        self.calls += 1
        return dspy.LMResponse.from_text(
            "usage",
            model=request.model,
            usage=dspy.core.LMUsage(input_tokens=3, output_tokens=2),
        )

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class FailOnceProvider(LMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        self.calls += 1
        if self.calls == 1:
            message = "temporary native failure"
            raise RuntimeError(message)
        return dspy.LMResponse.from_text("recovered", model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class RecordingCallback(BaseCallback):
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.last_exception: Exception | None = None

    def on_lm_start(
        self,
        call_id: str,
        instance: Any,  # noqa: ANN401
        inputs: dict[str, Any],
    ) -> None:
        _ = call_id, instance, inputs
        self.starts += 1

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        _ = call_id, outputs
        self.ends += 1
        self.last_exception = exception


def test_callbacks_history_and_usage_remain_owned_by_dspy() -> None:
    # Given a typed LM with a standard DSPy callback and usage-bearing response
    callback = RecordingCallback()
    lm = CustomLM(
        model="runtime/semantics",
        provider=UsageProvider(),
        callbacks=[callback],
        cache=False,
    )
    request = dspy.LMRequest.from_call(model=lm.model, prompt="measure")

    # When the public BaseLM call runs under DSPy's usage tracker
    with dspy.track_usage() as tracker:
        response = lm(request)

    # Then inherited DSPy machinery records each concern exactly once
    assert isinstance(response, dspy.LMResponse)
    assert callback.starts == callback.ends == 1
    assert callback.last_exception is None
    assert len(lm.history) == 1
    assert tracker.get_total_tokens()["runtime/semantics"]["total_tokens"] == 5


def test_disabled_history_uses_dspy_context_without_custom_branching() -> None:
    # Given a typed LM and explicit request
    lm = CustomLM(model="runtime/no-history", provider=UsageProvider(), cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="private")

    # When DSPy's native disabled-history context is active
    with dspy.context(disable_history=True):
        response = lm(request)

    # Then the response succeeds without a package-owned history implementation
    assert isinstance(response, dspy.LMResponse)
    assert lm.history == []


def test_failures_are_not_cached() -> None:
    # Given a cache-enabled LM whose provider fails before returning a response
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    provider = FailOnceProvider()
    lm = CustomLM(model="runtime/fail-once", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="retry later", cache=True)

    # When the failed request is attempted again
    with pytest.raises(dspy.LMUnexpectedError):
        lm.forward(request)
    recovered = lm.forward(request)

    # Then only the completed response enters DSPy's cache
    assert recovered.text == "recovered"
    assert provider.calls == 2


def test_callback_observes_the_final_normalized_failure_once() -> None:
    # Given a provider failure and a standard DSPy callback
    callback = RecordingCallback()
    lm = CustomLM(
        model="runtime/callback-failure",
        provider=FailOnceProvider(),
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


def test_cache_hits_clear_billed_usage_through_dspys_cache_policy() -> None:
    # Given a cache-enabled provider response with billed usage
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    provider = UsageProvider()
    lm = CustomLM(model="runtime/cached-usage", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="usage once", cache=True)

    # When the same request is completed twice
    first = lm.forward(request)
    cached = lm.forward(request)

    # Then the live response retains usage and the cache hit cannot be billed again
    assert first.usage_as_dict()["total_tokens"] == 5
    assert cached.usage_as_dict() == {}
    assert cached.cache_hit is True
    assert provider.calls == 1


def test_restricted_disk_cache_uses_dspys_safe_type_policy(tmp_path: Path) -> None:
    # Given DSPy's restricted cache without registered normalized response types
    unregistered = UsageProvider()
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path / "unregistered"),
        restrict_pickle=True,
    )
    lm = CustomLM(model="runtime/restricted-miss", provider=unregistered)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="safe types", cache=True)

    # When the unregistered value is read, then the public DSPy types are registered
    first = lm.forward(request)
    miss = lm.forward(request)
    registered = UsageProvider()
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path / "registered"),
        restrict_pickle=True,
        safe_types=[
            dspy.core.LMResponse,
            dspy.core.LMOutput,
            dspy.core.LMTextPart,
            dspy.core.LMUsage,
        ],
    )
    registered_lm = CustomLM(model="runtime/restricted-hit", provider=registered)
    registered_request = dspy.LMRequest.from_call(
        model=registered_lm.model,
        prompt="safe types",
        cache=True,
    )
    stored = registered_lm.forward(registered_request)
    hit = registered_lm.forward(registered_request)

    # Then unregistered data follows native cache-miss behavior and registered data round-trips
    assert first.cache_hit is miss.cache_hit is False
    assert unregistered.calls == 2
    assert stored.cache_hit is False
    assert hit.cache_hit is True
    assert registered.calls == 1
