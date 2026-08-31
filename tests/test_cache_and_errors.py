from __future__ import annotations

import anyio
import dspy
import pytest

from dspy_base_lm import CustomLM, LMProvider


class CountingProvider(LMProvider):
    """A deterministic provider whose call count exposes cache behavior."""

    def __init__(self) -> None:
        super().__init__()
        self.sync_calls = 0
        self.async_calls = 0
        self.retry_budgets: list[int] = []

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        self.sync_calls += 1
        self.retry_budgets.append(num_retries)
        return dspy.LMResponse.from_text(f"sync-{self.sync_calls}", model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        self.async_calls += 1
        self.retry_budgets.append(num_retries)
        return dspy.LMResponse.from_text(f"async-{self.async_calls}", model=request.model)


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
        msg = "native SDK failure"
        raise RuntimeError(msg)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class LegacyShapeProvider(LMProvider):
    """Deliberately violate the typed result contract to exercise rejection."""

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = request, num_retries
        return {"choices": []}

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class UnsafeResponseProvider(CountingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.runtime_client = object()

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        self.sync_calls += 1
        self.retry_budgets.append(num_retries)
        return dspy.LMResponse.from_text(
            f"unsafe-{self.sync_calls}",
            model=request.model,
            provider_response=self.runtime_client,
            provider_data={"authorization": "Bearer provider-secret"},
        )


class ConfiguredProvider(CountingProvider):
    def __init__(self, deployment: str) -> None:
        super().__init__()
        self.deployment = deployment

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        self.sync_calls += 1
        self.retry_budgets.append(num_retries)
        return dspy.LMResponse.from_text(self.deployment, model=request.model)


class ArbitraryMetadataProvider(CountingProvider):
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        self.sync_calls += 1
        self.retry_budgets.append(num_retries)
        return dspy.LMResponse.from_text(
            f"metadata-{self.sync_calls}",
            model=request.model,
            provider_response={"token": "tokenizer-output"},
        )


def _memory_cache() -> None:
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)


def test_cache_reuses_only_equivalent_completed_responses() -> None:
    # Given a memory cache and one provider-backed LM
    _memory_cache()
    provider = CountingProvider()
    lm = CustomLM(model="counting/cache", provider=provider, num_retries=7)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same", cache=True)

    # When the same typed request is completed twice
    first = lm.forward(request)
    second = lm.forward(request)

    # Then the provider is called once and DSPy marks the cached response
    assert first.text == "sync-1"
    assert second.text == "sync-1"
    assert second.cache_hit is True
    assert provider.sync_calls == 1
    assert provider.retry_budgets == [7]


def test_cache_identity_preserves_rollout_and_ignores_cache_toggle() -> None:
    # Given equivalent requests with different cache flags and rollout identities
    _memory_cache()
    provider = CountingProvider()
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


def test_cache_identity_distinguishes_behavior_changing_typed_config() -> None:
    # Given two otherwise equivalent requests with different generation behavior
    _memory_cache()
    provider = CountingProvider()
    lm = CustomLM(model="counting/config", provider=provider)
    conservative = dspy.LMRequest.from_call(model=lm.model, prompt="same", top_p=0.2)
    exploratory = dspy.LMRequest.from_call(model=lm.model, prompt="same", top_p=0.9)

    # When each typed request is repeated
    first_conservative = lm.forward(conservative)
    first_exploratory = lm.forward(exploratory)
    cached_conservative = lm.forward(conservative)
    cached_exploratory = lm.forward(exploratory)

    # Then each behavior-changing configuration owns a distinct cache entry
    assert first_conservative.text == cached_conservative.text == "sync-1"
    assert first_exploratory.text == cached_exploratory.text == "sync-2"
    assert cached_conservative.cache_hit is True
    assert cached_exploratory.cache_hit is True
    assert provider.sync_calls == 2


def test_cache_reuses_entries_across_equivalent_provider_instances() -> None:
    # Given equivalent providers addressed by the same stable model identity
    _memory_cache()
    first_provider = ConfiguredProvider("deployment-a")
    second_provider = ConfiguredProvider("deployment-a")
    first_lm = CustomLM(model="deployment-a/model", provider=first_provider)
    second_lm = CustomLM(model="deployment-a/model", provider=second_provider)
    request = dspy.LMRequest.from_call(model="deployment-a/model", prompt="same")

    # When both providers complete the equivalent request
    first = first_lm.forward(request)
    second = second_lm.forward(request)

    # Then DSPy's cache remains reusable beyond one runtime provider instance
    assert first.text == second.text == "deployment-a"
    assert second.cache_hit is True
    assert first_provider.sync_calls == 1
    assert second_provider.sync_calls == 0


def test_provider_deployments_use_distinct_model_identities() -> None:
    # Given two deployments owned by one provider implementation
    _memory_cache()
    first_provider = ConfiguredProvider("deployment-a")
    second_provider = ConfiguredProvider("deployment-b")
    first_lm = CustomLM(model="deployment-a/model", provider=first_provider)
    second_lm = CustomLM(model="deployment-b/model", provider=second_provider)

    # When each deployment receives an otherwise equivalent request
    first = first_lm.forward(
        dspy.LMRequest.from_call(model=first_lm.model, prompt="same"),
    )
    second = second_lm.forward(
        dspy.LMRequest.from_call(model=second_lm.model, prompt="same"),
    )

    # Then provider-qualified model identities prevent cross-deployment reuse
    assert first.text == "deployment-a"
    assert second.text == "deployment-b"
    assert first_provider.sync_calls == second_provider.sync_calls == 1


def test_cache_identity_includes_request_metadata() -> None:
    # Given requests whose safe metadata can affect provider behavior
    _memory_cache()
    provider = CountingProvider()
    lm = CustomLM(model="safe/cache", provider=provider)
    first = dspy.LMRequest.from_call(model=lm.model, prompt="same")
    first.metadata["tenant"] = "first"
    second = dspy.LMRequest.from_call(model=lm.model, prompt="same")
    second.metadata["tenant"] = "second"

    # When both requests use DSPy's cache
    first_response = lm.forward(first)
    second_response = lm.forward(second)

    # Then distinct metadata cannot return another request's cached response
    assert first_response.text == "sync-1"
    assert second_response.text == "sync-2"
    assert second_response.cache_hit is False
    assert provider.sync_calls == 2


def test_cache_accepts_arbitrary_json_metadata_keys() -> None:
    # Given a legitimate domain field whose name resembles runtime terminology
    _memory_cache()
    provider = CountingProvider()
    lm = CustomLM(model="safe/request", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same")
    request.messages[0].metadata["session"] = "conversation-7"

    # When the request crosses the LM boundary twice
    first = lm.forward(request)
    second = lm.forward(request)

    # Then key-name guesses do not reject or disable caching of valid JSON data
    assert first.text == second.text == "sync-1"
    assert second.cache_hit is True
    assert provider.sync_calls == 1


@pytest.mark.parametrize(
    "location",
    ["extensions", "metadata", "message_metadata", "response_format"],
)
def test_request_runtime_objects_are_rejected_before_provider_or_cache(location: str) -> None:
    # Given a runtime object hidden behind an innocuous request key
    _memory_cache()
    provider = CountingProvider()
    lm = CustomLM(model="safe/request-runtime", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same")
    if location == "extensions":
        request.config.extensions["backend"] = object()
    elif location == "metadata":
        request.metadata["backend"] = object()
    elif location == "message_metadata":
        request.messages[0].metadata["backend"] = object()
    else:
        request.config.response_format = object()

    # When the request crosses the LM boundary
    with pytest.raises(dspy.LMConfigurationError, match="JSON-like"):
        lm.forward(request)

    # Then validation runs before provider dispatch or cache identity creation
    assert provider.sync_calls == 0


def test_cyclic_request_metadata_raises_a_typed_configuration_error() -> None:
    # Given request metadata containing a reference cycle
    provider = CountingProvider()
    lm = CustomLM(model="safe/request-cycle", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same")
    request.metadata["nested"] = request.metadata

    # When the request crosses the LM boundary
    with pytest.raises(dspy.LMConfigurationError, match="JSON-like"):
        lm.forward(request)

    # Then no recursion failure, provider call, or cache write escapes
    assert provider.sync_calls == 0


def test_runtime_response_state_bypasses_dspys_cache() -> None:
    # Given a completed response containing a runtime client and credential metadata
    _memory_cache()
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


def test_arbitrary_json_response_keys_remain_cacheable() -> None:
    # Given a finite response containing a legitimate domain field
    _memory_cache()
    provider = ArbitraryMetadataProvider()
    lm = CustomLM(model="safe/arbitrary-response", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same")

    # When the same request completes twice
    first = lm.forward(request)
    second = lm.forward(request)

    # Then only structural serializability controls caching
    assert first.text == second.text == "metadata-1"
    assert second.cache_hit is True
    assert provider.sync_calls == 1


def test_async_cache_reuses_completed_response() -> None:
    # Given a typed asynchronous provider and memory cache
    _memory_cache()
    provider = CountingProvider()
    lm = CustomLM(model="counting/async-cache", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="same", cache=True)

    async def complete_twice() -> tuple[dspy.LMResponse, dspy.LMResponse]:
        # When the same asynchronous request is completed twice
        return await lm.aforward(request), await lm.aforward(request)

    first, second = anyio.run(complete_twice)

    # Then only the completed response is reused
    assert first.text == second.text == "async-1"
    assert second.cache_hit is True
    assert provider.async_calls == 1


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


def test_legacy_provider_response_shapes_are_rejected() -> None:
    # Given a provider that returns an obsolete completion dictionary
    lm = CustomLM(model="errors/legacy-shape", provider=LegacyShapeProvider(), cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="reject")

    # When the value crosses DSPy's typed public boundary
    with pytest.raises(TypeError, match="LMResponse"):
        _ = lm(request)

    # Then DSPy's own contract validation rejects it; CustomLM adds no parallel check
