from __future__ import annotations

import dspy

from dspy_base_lm import CustomLM, LMProvider


class CapturingProvider(LMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.request: dspy.LMRequest | None = None

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        self.request = request
        return dspy.LMResponse.from_text("captured", model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


def test_dspy_normalizes_rich_direct_inputs_before_provider_dispatch() -> None:
    # Given DSPy's typed roles, prior response, tool result, media parts, and tool schema
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
    assert provider.request is not None
    assert [message.role for message in provider.request.messages] == [
        "system",
        "assistant",
        "tool",
        "user",
    ]
    assert provider.request.messages[1].text == "previous answer"
    assert isinstance(provider.request.messages[2].parts[0], dspy.core.LMToolResultPart)
    user_parts = provider.request.messages[3].parts
    assert any(isinstance(part, dspy.core.LMImagePart) for part in user_parts)
    assert any(isinstance(part, dspy.core.LMAudioPart) for part in user_parts)
    assert any(isinstance(part, dspy.core.LMDocumentPart) for part in user_parts)
    assert provider.request.tools == [tool]
