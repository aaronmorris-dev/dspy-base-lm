from __future__ import annotations

import dspy

from dspy_base_lm import CustomLM, LMProvider


class RichResponseProvider(LMProvider):
    """Return representative normalized fields without transport coercion."""

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

    def __init__(self) -> None:
        super().__init__()
        self.native_response = {"runtime": "excluded"}

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


def test_normalized_response_fidelity_is_preserved() -> None:
    provider = RichResponseProvider()
    lm = CustomLM(model="rich/model", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="use every field")

    response = lm.forward(request)

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
    lm = CustomLM(model="rich/capabilities", provider=RichResponseProvider())

    capabilities = (
        lm.supports_function_calling,
        lm.supports_reasoning,
        lm.supports_response_schema,
        lm.supported_params,
    )

    assert capabilities == (True, True, True, {"reasoning", "response_format", "tools"})
