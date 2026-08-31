from __future__ import annotations

import json

import dspy
import pydantic

from dspy_base_lm import CustomLM, LMProvider


class AdapterProvider(LMProvider):
    """Return deterministic output in the format requested by DSPy's adapter."""

    def supported_params(self, model: str) -> frozenset[str]:
        _ = model
        return frozenset({"response_format"})

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        prompt = "\n".join(message.text or "" for message in request.messages)
        has_reasoning = "reasoning" in prompt.lower()
        if request.config.response_format is not None:
            payload = {"answer": "typed answer"}
            if has_reasoning:
                payload = {"reasoning": "typed reasoning", **payload}
            text = json.dumps(payload)
        elif has_reasoning:
            text = (
                "[[ ## reasoning ## ]]\ntyped reasoning\n\n"
                "[[ ## answer ## ]]\ntyped answer\n\n[[ ## completed ## ]]"
            )
        else:
            text = "[[ ## answer ## ]]\ntyped answer\n\n[[ ## completed ## ]]"
        return dspy.LMResponse.from_text(text, model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class SchemaProvider(LMProvider):
    """Declare native response-schema support and capture the typed request."""

    def __init__(self) -> None:
        super().__init__()
        self.last_request: dspy.LMRequest | None = None

    def supported_params(self, model: str) -> frozenset[str]:
        _ = model
        return frozenset({"response_format"})

    def supports_response_schema(self, model: str) -> bool:
        _ = model
        return True

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        self.last_request = request
        text = json.dumps({"answer": "typed answer"})
        return dspy.LMResponse.from_text(text, model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


def test_custom_lm_runs_through_predict_and_chat_adapter() -> None:
    # Given a deterministic typed LM configured with ChatAdapter
    lm = CustomLM(model="adapter/chat", provider=AdapterProvider(), cache=False)
    predict = dspy.Predict("question -> answer")

    # When a real DSPy module invokes the LM
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = predict(question="Does the typed boundary work?")

    # Then the adapter parses the provider's answer
    assert result.answer == "typed answer"
    assert len(lm.history) == 1
    assert lm.history[0]["model"] == "adapter/chat"
    assert lm.history[0]["response_model"] == "adapter/chat"


def test_custom_lm_runs_through_chain_of_thought() -> None:
    # Given a deterministic typed LM and ChainOfThought module
    lm = CustomLM(model="adapter/reasoning", provider=AdapterProvider(), cache=False)
    program = dspy.ChainOfThought("question -> answer")

    # When the module requests reasoning and an answer
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = program(question="Why use typed requests?")

    # Then both signature fields survive the DSPy adapter path
    assert result.reasoning == "typed reasoning"
    assert result.answer == "typed answer"


def test_custom_lm_runs_through_json_adapter() -> None:
    # Given a provider that declares native response-format support
    lm = CustomLM(model="adapter/json", provider=AdapterProvider(), cache=False)
    predict = dspy.Predict("question -> answer")

    # When JSONAdapter selects the typed response format
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        result = predict(question="Return structured output")

    # Then DSPy parses the normalized provider response
    assert result.answer == "typed answer"
    assert lm.supports_response_schema is False
    assert lm.supported_params == {"response_format"}


def test_json_adapter_native_structured_output_reaches_the_provider() -> None:
    # Given a provider that declares native response-schema support
    provider = SchemaProvider()
    lm = CustomLM(model="adapter/schema", provider=provider, cache=False)

    # When JSONAdapter selects its native structured-output path
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        result = dspy.Predict("question -> answer")(question="Return structured output")

    # Then the declarative Pydantic response format reaches the provider untranslated
    assert result.answer == "typed answer"
    assert provider.last_request is not None
    response_format = provider.last_request.config.response_format
    assert isinstance(response_format, type)
    assert issubclass(response_format, pydantic.BaseModel)
