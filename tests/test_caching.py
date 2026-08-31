"""DSPy cache behavior: reuse, identity, and storage safety."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
import dspy
import pydantic
import pytest

from conftest import CountingProvider
from dspy_base_lm import CustomLM

if TYPE_CHECKING:
    from pathlib import Path


class UnsafeResponseProvider(CountingProvider):
    """Attach a runtime client to the response, which must never be cached."""

    def __init__(self) -> None:
        super().__init__()
        self.runtime_client = object()

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        self.sync_calls += 1
        return dspy.LMResponse.from_text(
            f"unsafe-{self.sync_calls}",
            model=request.model,
            provider_response=self.runtime_client,
            provider_data={"authorization": "Bearer provider-secret"},
        )


class JsonResponseProvider(CountingProvider):
    """Attach finite JSON provider data, which stays cacheable."""

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        self.sync_calls += 1
        return dspy.LMResponse.from_text(
            f"metadata-{self.sync_calls}",
            model=request.model,
            provider_response={"token": "tokenizer-output"},
        )


class FailOnceProvider(CountingProvider):
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        self.sync_calls += 1
        if self.sync_calls == 1:
            message = "temporary native failure"
            raise RuntimeError(message)
        return dspy.LMResponse.from_text("recovered", model=request.model)


def test_repeated_requests_reuse_the_cached_response(
    memory_cache: None,
    counting_provider: CountingProvider,
) -> None:
    # Given a memory cache and one provider-backed LM
    provider = counting_provider
    lm = CustomLM(model="counting/cache", provider=provider, num_retries=7)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same", cache=True)

    # When the same typed request is completed twice
    first = lm.forward(request)
    second = lm.forward(request)

    # Then the provider is called once with its retry budget, and DSPy marks the hit
    assert first.text == second.text == "sync-1"
    assert second.cache_hit is True
    assert provider.sync_calls == 1
    assert provider.retry_budgets == [7]


def test_async_requests_reuse_the_cached_response(
    memory_cache: None,
    counting_provider: CountingProvider,
) -> None:
    # Given a typed asynchronous provider and memory cache
    provider = counting_provider
    lm = CustomLM(model="counting/async-cache", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same", cache=True)

    async def complete_twice() -> tuple[dspy.LMResponse, dspy.LMResponse]:
        return await lm.aforward(request), await lm.aforward(request)

    first, second = anyio.run(complete_twice)

    # Then only the completed response is reused
    assert first.text == second.text == "async-1"
    assert second.cache_hit is True
    assert provider.async_calls == 1


def test_cache_identity_covers_behavioral_request_fields(
    memory_cache: None,
    counting_provider: CountingProvider,
) -> None:
    # Given requests that independently vary every behavior-affecting field
    provider = counting_provider
    lm = CustomLM(model="identity/model-a", provider=provider)
    tool = dspy.core.LMToolSpec(name="lookup", parameters={"type": "object"})
    with_metadata = dspy.LMRequest.from_call(model="identity/model-a", prompt="same")
    with_metadata.metadata["tenant"] = "first"
    requests = [
        dspy.LMRequest.from_call(model="identity/model-a", prompt="same"),
        dspy.LMRequest.from_call(model="identity/model-b", prompt="same"),
        dspy.LMRequest.from_call(model="identity/model-a", prompt="different"),
        dspy.LMRequest.from_call(model="identity/model-a", prompt="same", tools=[tool]),
        dspy.LMRequest.from_call(model="identity/model-a", prompt="same", top_p=0.9),
        dspy.LMRequest.from_call(
            model="identity/model-a",
            prompt="same",
            extensions={"region": "east"},
        ),
        with_metadata,
    ]

    # When every request is completed and then repeated
    first_responses = [lm.forward(request) for request in requests]
    cached_responses = [lm.forward(request) for request in requests]

    # Then each behavioral shape owns one entry and every exact repeat is a hit
    assert provider.sync_calls == len(requests)
    assert [response.text for response in first_responses] == [
        f"sync-{index}" for index in range(1, len(requests) + 1)
    ]
    assert all(response.cache_hit for response in cached_responses)


def test_cache_identity_preserves_rollout_and_ignores_cache_toggle(
    memory_cache: None,
    counting_provider: CountingProvider,
) -> None:
    # Given equivalent requests with different cache flags and rollout identities
    provider = counting_provider
    lm = CustomLM(model="counting/rollout", provider=provider)
    enabled = dspy.LMRequest.from_call(model=lm.model, prompt="same", cache=True, rollout_id=1)
    disabled = dspy.LMRequest.from_call(model=lm.model, prompt="same", cache=False, rollout_id=1)
    next_rollout = dspy.LMRequest.from_call(model=lm.model, prompt="same", cache=True, rollout_id=2)

    # When cache policy and rollout are varied
    first = lm.forward(enabled)
    uncached = lm.forward(disabled)
    same_rollout = lm.forward(enabled)
    different_rollout = lm.forward(next_rollout)

    # Then policy does not alter identity, disabled calls bypass storage, and rollout does
    assert first.text == "sync-1"
    assert uncached.text == "sync-2"
    assert same_rollout.text == "sync-1"
    assert different_rollout.text == "sync-3"
    assert provider.sync_calls == 3


def test_cache_entries_are_shared_by_model_identity_not_provider_instance(
    memory_cache: None,
) -> None:
    # Given equivalent and distinct deployments across provider instances
    first_provider = CountingProvider()
    second_provider = CountingProvider()
    other_provider = CountingProvider()
    first_lm = CustomLM(model="deployment-a/model", provider=first_provider)
    second_lm = CustomLM(model="deployment-a/model", provider=second_provider)
    other_lm = CustomLM(model="deployment-b/model", provider=other_provider)

    # When each LM completes an otherwise equivalent request
    first = first_lm.forward(dspy.LMRequest.from_call(model=first_lm.model, prompt="same"))
    second = second_lm.forward(dspy.LMRequest.from_call(model=second_lm.model, prompt="same"))
    other = other_lm.forward(dspy.LMRequest.from_call(model=other_lm.model, prompt="same"))

    # Then equivalent model identities share entries and distinct ones never do
    assert first.text == second.text == "sync-1"
    assert second.cache_hit is True
    assert second_provider.sync_calls == 0
    assert other.cache_hit is False
    assert other_provider.sync_calls == 1


def test_pydantic_response_format_requests_cache_by_schema(
    memory_cache: None,
    counting_provider: CountingProvider,
) -> None:
    # Given a request whose response_format is a Pydantic model class
    class Answer(pydantic.BaseModel):
        answer: str

    provider = counting_provider
    lm = CustomLM(model="identity/schema", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same", response_format=Answer)

    # When the request is completed and repeated
    first = lm.forward(request)
    second = lm.forward(request)

    # Then the declarative schema is a stable cache identity
    assert provider.sync_calls == 1
    assert first.text == "sync-1"
    assert second.cache_hit is True


def test_unsafe_response_values_bypass_cache_storage(memory_cache: None) -> None:
    # Given a completed response containing a runtime client and credential metadata
    provider = UnsafeResponseProvider()
    lm = CustomLM(model="safe/response", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same")

    # When the same request completes twice
    first = lm.forward(request)
    second = lm.forward(request)

    # Then each live response is returned but neither unsafe value is cached
    assert first.text == "unsafe-1"
    assert second.text == "unsafe-2"
    assert first.provider_response is second.provider_response is provider.runtime_client
    assert first.cache_hit is second.cache_hit is False
    assert provider.sync_calls == 2


def test_arbitrary_json_response_values_remain_cacheable(memory_cache: None) -> None:
    # Given a finite response containing a legitimate domain field
    provider = JsonResponseProvider()
    lm = CustomLM(model="safe/arbitrary-response", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same")

    # When the same request completes twice
    first = lm.forward(request)
    second = lm.forward(request)

    # Then only structural serializability controls caching
    assert first.text == second.text == "metadata-1"
    assert second.cache_hit is True
    assert provider.sync_calls == 1


def test_failed_calls_are_not_cached(memory_cache: None) -> None:
    # Given a cache-enabled LM whose provider fails before returning a response
    provider = FailOnceProvider()
    lm = CustomLM(model="runtime/fail-once", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="retry later", cache=True)

    # When the failed request is attempted again
    with pytest.raises(dspy.LMUnexpectedError):
        lm.forward(request)
    recovered = lm.forward(request)

    # Then only the completed response enters DSPy's cache
    assert recovered.text == "recovered"
    assert provider.sync_calls == 2


def test_cache_hits_clear_billed_usage(
    memory_cache: None,
    usage_provider: CountingProvider,
) -> None:
    # Given a cache-enabled provider response with billed usage
    provider = usage_provider
    lm = CustomLM(model="runtime/cached-usage", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="usage once", cache=True)

    # When the same request is completed twice
    first = lm.forward(request)
    cached = lm.forward(request)

    # Then the live response retains usage and the cache hit cannot be billed again
    assert first.usage_as_dict()["total_tokens"] == 5
    assert cached.usage_as_dict() == {}
    assert cached.cache_hit is True
    assert provider.sync_calls == 1


def test_restricted_disk_cache_follows_dspys_safe_type_policy(tmp_path: Path) -> None:
    # Given DSPy's restricted cache without registered normalized response types
    unregistered = CountingProvider(usage=dspy.core.LMUsage(input_tokens=3, output_tokens=2))
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
    registered = CountingProvider(usage=dspy.core.LMUsage(input_tokens=3, output_tokens=2))
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
    assert unregistered.sync_calls == 2
    assert stored.cache_hit is False
    assert hit.cache_hit is True
    assert registered.sync_calls == 1
