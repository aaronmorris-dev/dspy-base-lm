from __future__ import annotations

import anyio
import dspy
import pytest

import dspy_base_lm
from dspy_base_lm import CustomLM, LMProvider


class CapturingProvider(LMProvider):
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


class FalseyProvider(CapturingProvider):
    """A valid runtime provider whose SDK-style truth value is false."""

    def __bool__(self) -> bool:
        return False


def test_custom_lm_fails_when_no_provider_is_configured() -> None:
    # Given an otherwise valid custom model identifier
    # When no provider is injected or inferred
    # Then construction fails with DSPy's configuration error
    with pytest.raises(dspy.LMNotConfiguredError, match="provider"):
        CustomLM(model="example/model")


def test_injected_provider_is_selected_by_presence_not_truthiness() -> None:
    # Given a valid provider object with a false truth value
    provider = FalseyProvider()

    # When CustomLM receives the provider explicitly
    lm = CustomLM(model="example/falsey-provider", provider=provider)

    # Then it preserves that runtime object instead of attempting inference
    assert lm.provider is provider


def test_custom_lm_returns_a_typed_response() -> None:
    # Given a deterministic runtime provider
    provider = CapturingProvider()
    lm = CustomLM(model="test/capturing", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="hello DSPy")

    # When the typed provider boundary is called
    response = lm.forward(request)

    # Then it returns DSPy's response type without transport-specific coercion
    assert isinstance(response, dspy.LMResponse)
    assert response.text == "captured"
    assert response.model == "test/capturing"


def test_custom_lm_returns_a_typed_response_asynchronously() -> None:
    # Given a deterministic runtime provider
    lm = CustomLM(model="test/capturing", provider=CapturingProvider())
    request = dspy.LMRequest.from_call(model=lm.model, prompt="hello async DSPy")

    async def call_lm() -> dspy.LMResponse:
        return await lm.aforward(request)

    # When the typed asynchronous provider boundary is called
    response = anyio.run(call_lm)

    # Then it returns the same DSPy response contract
    assert isinstance(response, dspy.LMResponse)
    assert response.text == "captured"


def test_public_api_contains_only_the_reference_types() -> None:
    # Given the installed package API
    # When its explicit exports are inspected
    exports = set(dspy_base_lm.__all__)

    # Then only the intentional teaching surface is public
    assert exports == {"CustomLM", "LMProvider"}


def test_base_lm_configuration_flows_into_the_typed_request() -> None:
    # Given standard BaseLM defaults and provider extensions
    provider = CapturingProvider()
    lm = CustomLM(
        model="config/model",
        provider=provider,
        cache=False,
        top_p=0.8,
        extensions={"region": "local"},
    )

    # When DSPy's normalized direct-call path is used
    with dspy.context(experimental=True):
        response = lm(dspy.User("capture configuration"))

    # Then BaseLM owns normalization and the provider receives grouped typed config
    assert isinstance(response, dspy.LMResponse)
    assert provider.last_request is not None
    assert provider.last_request.config.top_p == 0.8
    assert provider.last_request.config.extensions == {"region": "local"}


def test_explicit_request_overrides_remain_owned_by_base_lm() -> None:
    # Given an explicit typed request and a provider-backed LM
    provider = CapturingProvider()
    lm = CustomLM(model="config/overrides", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="override", top_p=0.2)

    # When a public call supplies a per-call generation override
    response = lm(request, top_p=0.9)

    # Then BaseLM creates an updated request without mutating the caller's value
    assert isinstance(response, dspy.LMResponse)
    assert provider.last_request is not None
    assert provider.last_request.config.top_p == 0.9
    assert request.config.top_p == 0.2


def test_public_async_call_normalizes_and_finalizes_the_typed_request() -> None:
    # Given a provider-backed LM and an explicit typed request
    provider = CapturingProvider()
    lm = CustomLM(model="config/async-call", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="async public call")

    async def call_lm() -> dspy.LMResponse:
        response = await lm.acall(request, temperature=0.4)
        assert isinstance(response, dspy.LMResponse)
        return response

    # When DSPy's public async entry point is used
    response = anyio.run(call_lm)

    # Then the request is normalized and DSPy records the finalized typed response
    assert response.text == "captured"
    assert provider.last_request is not None
    assert provider.last_request.config.temperature == 0.4
    assert len(lm.history) == 1
