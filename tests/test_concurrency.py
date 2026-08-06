from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import anyio
import dspy

from dspy_base_lm import CustomLM, LMProvider


class ConcurrentProvider(LMProvider):
    def __init__(self) -> None:
        super().__init__()
        self._lock = Lock()
        self.seen_temperatures: list[float | None] = []

    def _record(self, request: dspy.LMRequest) -> dspy.LMResponse:
        temperature = request.config.temperature
        with self._lock:
            self.seen_temperatures.append(temperature)
        return dspy.LMResponse.from_text(str(temperature), model=request.model)

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        return self._record(request)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        _ = num_retries
        await anyio.lowlevel.checkpoint()
        return self._record(request)


def test_concurrent_public_calls_keep_request_overrides_isolated() -> None:
    # Given one provider shared by sync and async calls to the same request
    provider = ConcurrentProvider()
    lm = CustomLM(model="concurrency/model", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="run", temperature=0.0)
    sync_temperatures = [0.1, 0.2, 0.3, 0.4]
    async_temperatures = [0.5, 0.6, 0.7, 0.8]

    # When callers supply independent per-call overrides concurrently
    def sync_call(temperature: float) -> str | None:
        response = lm(request, temperature=temperature)
        assert isinstance(response, dspy.LMResponse)
        return response.text

    with ThreadPoolExecutor(max_workers=4) as executor:
        sync_results = list(executor.map(sync_call, sync_temperatures))

    async def async_calls() -> list[str | None]:
        results: list[str | None] = [None] * len(async_temperatures)

        async def call(index: int, temperature: float) -> None:
            response = await lm.acall(request, temperature=temperature)
            assert isinstance(response, dspy.LMResponse)
            results[index] = response.text

        async with anyio.create_task_group() as task_group:
            for index, temperature in enumerate(async_temperatures):
                task_group.start_soon(call, index, temperature)
        return results

    async_results = anyio.run(async_calls)

    # Then DSPy isolates each normalized copy while the provider protects its runtime state
    assert sorted(sync_results) == sorted(str(value) for value in sync_temperatures)
    assert sorted(async_results) == sorted(str(value) for value in async_temperatures)
    assert sorted(provider.seen_temperatures) == sorted(sync_temperatures + async_temperatures)
    assert request.config.temperature == 0.0
    assert len(lm.history) == 8
