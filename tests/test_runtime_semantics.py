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
        instance: Any,
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
    callback = RecordingCallback()
    lm = CustomLM(
        model="runtime/semantics",
        provider=UsageProvider(),
        callbacks=[callback],
        cache=False,
    )
    request = dspy.LMRequest.from_call(model=lm.model, prompt="measure")

    with dspy.track_usage() as tracker:
        response = lm(request)

    assert isinstance(response, dspy.LMResponse)
    assert callback.starts == callback.ends == 1
    assert callback.last_exception is None
    assert len(lm.history) == 1
    assert tracker.get_total_tokens()["runtime/semantics"]["total_tokens"] == 5


def test_disabled_history_uses_dspy_context_without_custom_branching() -> None:
    lm = CustomLM(model="runtime/no-history", provider=UsageProvider(), cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="private")

    with dspy.context(disable_history=True):
        response = lm(request)

    assert isinstance(response, dspy.LMResponse)
    assert lm.history == []


def test_failures_are_not_cached() -> None:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    provider = FailOnceProvider()
    lm = CustomLM(model="runtime/fail-once", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="retry later", cache=True)

    with pytest.raises(dspy.LMUnexpectedError):
        lm.forward(request)
    recovered = lm.forward(request)

    assert recovered.text == "recovered"
    assert provider.calls == 2


def test_callback_observes_the_final_normalized_failure_once() -> None:
    callback = RecordingCallback()
    lm = CustomLM(
        model="runtime/callback-failure",
        provider=FailOnceProvider(),
        callbacks=[callback],
        cache=False,
    )
    request = dspy.LMRequest.from_call(model=lm.model, prompt="fail once")

    with pytest.raises(dspy.LMUnexpectedError) as caught:
        lm(request)

    assert callback.starts == callback.ends == 1
    assert callback.last_exception is caught.value
    assert lm.history == []


def test_cache_hits_clear_billed_usage_through_dspys_cache_policy() -> None:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    provider = UsageProvider()
    lm = CustomLM(model="runtime/cached-usage", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="usage once", cache=True)

    first = lm.forward(request)
    cached = lm.forward(request)

    assert first.usage_as_dict()["total_tokens"] == 5
    assert cached.usage_as_dict() == {}
    assert cached.cache_hit is True
    assert provider.calls == 1


def test_restricted_disk_cache_uses_dspys_safe_type_policy(tmp_path: Path) -> None:
    unregistered = UsageProvider()
    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=False,
        disk_cache_dir=str(tmp_path / "unregistered"),
        restrict_pickle=True,
    )
    lm = CustomLM(model="runtime/restricted-miss", provider=unregistered)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="safe types", cache=True)

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

    assert first.cache_hit is miss.cache_hit is False
    assert unregistered.calls == 2
    assert stored.cache_hit is False
    assert hit.cache_hit is True
    assert registered.calls == 1
