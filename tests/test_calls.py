"""The public call contract: construction, dispatch, fidelity, and runtime semantics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import anyio
import dspy
import pytest

import dspy_base_lm
from conftest import CapturingProvider, CountingProvider, RecordingCallback
from dspy_base_lm import CustomLM, LMProvider


class FalseBooleanProvider(CapturingProvider):
    """A valid runtime provider whose boolean value is false, like some SDK clients."""

    def __bool__(self) -> bool:
        return False


class RichResponseProvider(LMProvider):
    """Return representative normalized fields without transport coercion."""

    def __init__(self) -> None:
        super().__init__()
        self.native_response = {"runtime": "excluded"}

    def supports_function_calling(self, model: str) -> bool:
        return model.endswith("capabilities")

    def supports_reasoning(self, model: str) -> bool:
        return model.endswith("capabilities")

    def supports_response_schema(self, model: str) -> bool:
        return model.endswith("capabilities")

    def supported_params(self, model: str) -> frozenset[str]:
        if model.endswith("capabilities"):
            return frozenset({"reasoning", "response_format", "tools"})
        return frozenset()

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        return dspy.LMResponse(
            model=request.model,
            outputs=[
                dspy.core.LMOutput(
                    parts=[
                        dspy.core.LMThinkingPart(text="reasoning"),
                        dspy.core.LMTextPart(text="primary"),
                        dspy.core.LMToolCallPart(id="call-1", name="lookup", args={"q": "DSPy"}),
                        dspy.core.LMCitationPart(title="DSPy", url="https://dspy.ai"),
                        dspy.core.LMImagePart(url="https://example.test/image.png"),
                        dspy.core.LMAudioPart(data="YXVkaW8="),
                        dspy.core.LMDocumentPart(url="https://example.test/report.pdf"),
                    ],
                    finish_reason="tool_use",
                    truncated=True,
                    logprobs={"token": -0.1},
                    provider_data={"candidate": 0},
                ),
                dspy.core.LMOutput(
                    parts=[dspy.core.LMRefusalPart(text="not available")],
                    finish_reason="stop",
                ),
            ],
            usage=dspy.core.LMUsage(input_tokens=4, output_tokens=6, reasoning_tokens=2),
            cost=0.01,
            response_id="response-1",
            provider_response=self.native_response,
            provider_data={"region": "local"},
            metadata={"trace": "safe"},
        )

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


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


def test_provider_must_be_injected_or_inferred() -> None:
    # Given no injected provider, construction fails with DSPy's configuration error
    with pytest.raises(dspy.LMNotConfiguredError, match="provider"):
        CustomLM(model="example/model")

    # And a provider whose boolean value is false is still selected by presence
    provider = FalseBooleanProvider()
    lm = CustomLM(model="example/false-boolean-provider", provider=provider)
    assert lm.provider is provider


def test_forward_returns_typed_responses_sync_and_async(
    capturing_provider: CapturingProvider,
) -> None:
    # Given a deterministic runtime provider
    lm = CustomLM(model="test/capturing", provider=capturing_provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="hello DSPy")

    # When both typed provider boundaries are called
    response = lm.forward(request)

    async def call_lm() -> dspy.LMResponse:
        return await lm.aforward(request)

    async_response = anyio.run(call_lm)

    # Then both return DSPy's response type without transport-specific coercion
    assert isinstance(response, dspy.LMResponse)
    assert isinstance(async_response, dspy.LMResponse)
    assert response.text == async_response.text == "captured"
    assert response.model == "test/capturing"


def test_public_api_exports_custom_lm_and_lm_provider_only() -> None:
    assert set(dspy_base_lm.__all__) == {"CustomLM", "LMProvider"}


def test_base_lm_configuration_flows_into_the_typed_request(
    capturing_provider: CapturingProvider,
) -> None:
    # Given standard BaseLM defaults and provider extensions
    provider = capturing_provider
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


def test_explicit_request_overrides_remain_owned_by_base_lm(
    capturing_provider: CapturingProvider,
) -> None:
    # Given an explicit typed request and a provider-backed LM
    provider = capturing_provider
    lm = CustomLM(model="config/overrides", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="override", top_p=0.2)

    # When a public call supplies a per-call generation override
    response = lm(request, top_p=0.9)

    # Then BaseLM creates an updated request without mutating the caller's value
    assert isinstance(response, dspy.LMResponse)
    assert provider.last_request is not None
    assert provider.last_request.config.top_p == 0.9
    assert request.config.top_p == 0.2


def test_rich_direct_inputs_are_normalized_before_provider_dispatch(
    capturing_provider: CapturingProvider,
) -> None:
    # Given DSPy's typed roles, prior response, tool result, media parts, and tool schema
    provider = capturing_provider
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

    # When the public typed call path renders those values
    with dspy.context(experimental=True):
        response = lm(
            dspy.System("Follow the evidence."),
            prior,
            dspy.ToolResult(call_id="call-1", name="lookup", content="tool output"),
            dspy.User("Inspect these inputs.", image, audio, document),
            tools=[tool],
        )

    # Then CustomLM passes DSPy's complete typed request through unchanged
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


def test_rich_response_fields_are_preserved() -> None:
    # Given a provider that returns rich DSPy-native output
    provider = RichResponseProvider()
    lm = CustomLM(model="rich/model", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="use every field")

    # When CustomLM dispatches the normalized request
    response = lm.forward(request)

    # Then no typed response field is flattened or rebuilt
    assert len(response.outputs) == 2
    assert response.text == "primary"
    assert response.reasoning_content == "reasoning"
    assert response.tool_calls[0].name == "lookup"
    assert response.citations[0].title == "DSPy"
    assert response.images[0].url == "https://example.test/image.png"
    assert response.audio[0].data == "YXVkaW8="
    assert response.documents[0].url == "https://example.test/report.pdf"
    assert response.outputs[1].refusal == "not available"
    assert response.output.finish_reason == "tool_use"
    assert response.output.truncated is True
    assert response.output.logprobs == {"token": -0.1}
    assert response.response_id == "response-1"
    assert response.provider_response is provider.native_response
    assert response.provider_data == {"region": "local"}
    assert response.metadata == {"trace": "safe"}
    assert response.usage is not None
    assert response.usage.total_tokens == 10


def test_capabilities_delegate_to_the_provider() -> None:
    # Given a provider that declares only capabilities it actually implements
    lm = CustomLM(model="rich/capabilities", provider=RichResponseProvider())

    # Then CustomLM delegates without a parallel capability model
    assert lm.supports_function_calling is True
    assert lm.supports_reasoning is True
    assert lm.supports_response_schema is True
    assert lm.supported_params == {"reasoning", "response_format", "tools"}


def test_callbacks_history_and_usage_remain_owned_by_dspy(
    usage_provider: CountingProvider,
    recording_callback: RecordingCallback,
) -> None:
    # Given a typed LM with a standard DSPy callback and usage-bearing response
    callback = recording_callback
    lm = CustomLM(
        model="runtime/semantics",
        provider=usage_provider,
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


def test_disabled_history_uses_dspys_native_context(
    counting_provider: CountingProvider,
) -> None:
    # Given a typed LM and explicit request
    lm = CustomLM(model="runtime/no-history", provider=counting_provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="private")

    # When DSPy's native disabled-history context is active
    with dspy.context(disable_history=True):
        response = lm(request)

    # Then the response succeeds without a package-owned history implementation
    assert isinstance(response, dspy.LMResponse)
    assert lm.history == []


def test_concurrent_calls_keep_request_overrides_isolated() -> None:
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

    # Then DSPy isolates each normalized copy while the provider protects its state
    assert sorted(sync_results) == sorted(str(value) for value in sync_temperatures)
    assert sorted(async_results) == sorted(str(value) for value in async_temperatures)
    assert sorted(provider.seen_temperatures) == sorted(sync_temperatures + async_temperatures)
    assert request.config.temperature == 0.0
    assert len(lm.history) == 8
