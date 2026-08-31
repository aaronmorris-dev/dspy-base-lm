"""Shared providers, callbacks, and fixtures for the test suite.

Single-instance needs are served by fixtures; tests that construct several
providers or subclass one import the classes from this module directly.
"""

from __future__ import annotations

from typing import Any

import dspy
import pytest
from dspy.utils.callback import BaseCallback

from dspy_base_lm import LMProvider


class CountingProvider(LMProvider):
    """Deterministic provider whose call counts expose cache and retry behavior."""

    def __init__(self, usage: dspy.core.LMUsage | None = None) -> None:
        super().__init__()
        self.usage = usage
        self.sync_calls = 0
        self.async_calls = 0
        self.retry_budgets: list[int] = []

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        self.sync_calls += 1
        self.retry_budgets.append(num_retries)
        return dspy.LMResponse.from_text(
            f"sync-{self.sync_calls}",
            model=request.model,
            usage=self.usage,
        )

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        self.async_calls += 1
        self.retry_budgets.append(num_retries)
        return dspy.LMResponse.from_text(
            f"async-{self.async_calls}",
            model=request.model,
            usage=self.usage,
        )


class CapturingProvider(LMProvider):
    """Capture the last typed request and return deterministic text."""

    def __init__(self) -> None:
        super().__init__()
        self.last_request: dspy.LMRequest | None = None

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        self.last_request = request
        return dspy.LMResponse.from_text("captured", model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class RecordingCallback(BaseCallback):
    """Record DSPy callback invocations and the final exception, if any."""

    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0
        self.last_exception: Exception | None = None

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
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


@pytest.fixture
def memory_cache() -> None:
    """Configure DSPy's in-memory cache for tests that assert cache behavior."""
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)


@pytest.fixture
def counting_provider() -> CountingProvider:
    return CountingProvider()


@pytest.fixture
def usage_provider() -> CountingProvider:
    return CountingProvider(usage=dspy.core.LMUsage(input_tokens=3, output_tokens=2))


@pytest.fixture
def capturing_provider() -> CapturingProvider:
    return CapturingProvider()


@pytest.fixture
def recording_callback() -> RecordingCallback:
    return RecordingCallback()
