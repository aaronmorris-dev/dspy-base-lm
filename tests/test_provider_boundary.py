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
    with pytest.raises(dspy.LMNotConfiguredError, match="provider"):
        CustomLM(model="example/model")


def test_injected_provider_is_selected_by_presence_not_truthiness() -> None:
    provider = FalseyProvider()

    lm = CustomLM(model="example/falsey-provider", provider=provider)

    assert lm.provider is provider


def test_custom_lm_returns_a_typed_response() -> None:
    provider = CapturingProvider()
    lm = CustomLM(model="test/capturing", provider=provider)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="hello DSPy")

    response = lm.forward(request)

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "captured"
    assert response.model == "test/capturing"


def test_custom_lm_returns_a_typed_response_asynchronously() -> None:
    lm = CustomLM(model="test/capturing", provider=CapturingProvider())
    request = dspy.LMRequest.from_call(model=lm.model, prompt="hello async DSPy")

    async def call_lm() -> dspy.LMResponse:
        return await lm.aforward(request)

    response = anyio.run(call_lm)

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "captured"


def test_public_api_contains_only_the_reference_types() -> None:
    exports = set(dspy_base_lm.__all__)

    assert exports == {"CustomLM", "LMProvider"}


def test_base_lm_configuration_flows_into_the_typed_request() -> None:
    provider = CapturingProvider()
    lm = CustomLM(
        model="config/model",
        provider=provider,
        cache=False,
        top_p=0.8,
        extensions={"region": "local"},
    )

    with dspy.context(experimental=True):
        response = lm(dspy.User("capture configuration"))

    assert isinstance(response, dspy.LMResponse)
    assert provider.last_request is not None
    assert provider.last_request.config.top_p == 0.8
    assert provider.last_request.config.extensions == {"region": "local"}


def test_explicit_request_overrides_remain_owned_by_base_lm() -> None:
    provider = CapturingProvider()
    lm = CustomLM(model="config/overrides", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="override", top_p=0.2)

    response = lm(request, top_p=0.9)

    assert isinstance(response, dspy.LMResponse)
    assert provider.last_request is not None
    assert provider.last_request.config.top_p == 0.9
    assert request.config.top_p == 0.2


def test_public_async_call_normalizes_and_finalizes_the_typed_request() -> None:
    provider = CapturingProvider()
    lm = CustomLM(model="config/async-call", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="async public call")

    async def call_lm() -> dspy.LMResponse:
        response = await lm.acall(request, temperature=0.4)
        assert isinstance(response, dspy.LMResponse)
        return response

    response = anyio.run(call_lm)

    assert response.text == "captured"
    assert provider.last_request is not None
    assert provider.last_request.config.temperature == 0.4
    assert len(lm.history) == 1


def test_dspy_normalizes_rich_direct_inputs_before_provider_dispatch() -> None:
    provider = CapturingProvider()
    lm = CustomLM(model="fidelity/request", provider=provider, cache=False)
    prior = dspy.LMResponse.from_text("previous answer", model=lm.model)
    image = dspy.core.LMImagePart(url="https://example.test/image.png")
    audio = dspy.core.LMAudioPart(data="YXVkaW8=")
    document = dspy.core.LMDocumentPart(url="https://example.test/report.pdf")
    tool = dspy.core.LMToolSpec(
        name="lookup",
        description="Look up a topic",
        parameters={"type": "object", "properties": {"topic": {"type": "string"}}},
    )

    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Follow the evidence."),
            prior,
            dspy.ToolResult(call_id="call-1", name="lookup", content="tool output"),
            dspy.User("Inspect these inputs.", image, audio, document),
            tools=[tool],
        )

    assert isinstance(response, dspy.LMResponse)
    assert provider.last_request is not None
    assert [message.role for message in provider.last_request.messages] == [
        "system",
        "assistant",
        "tool",
        "user",
    ]
    assert provider.last_request.messages[1].text == "previous answer"
    assert isinstance(provider.last_request.messages[2].parts[0], dspy.core.LMToolResultPart)
    user_parts = provider.last_request.messages[3].parts
    assert any(isinstance(part, dspy.core.LMImagePart) for part in user_parts)
    assert any(isinstance(part, dspy.core.LMAudioPart) for part in user_parts)
    assert any(isinstance(part, dspy.core.LMDocumentPart) for part in user_parts)
    assert provider.last_request.tools == [tool]
