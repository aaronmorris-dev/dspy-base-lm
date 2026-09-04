from __future__ import annotations

import json

import dspy

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


def test_custom_lm_runs_through_predict_and_chat_adapter() -> None:
    lm = CustomLM(model="adapter/chat", provider=AdapterProvider(), cache=False)
    predict = dspy.Predict("question -> answer")

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = predict(question="Does the typed boundary work?")

    assert result.answer == "typed answer"
    assert len(lm.history) == 1
    assert lm.history[0]["model"] == "adapter/chat"
    assert lm.history[0]["response_model"] == "adapter/chat"


def test_custom_lm_runs_through_chain_of_thought() -> None:
    lm = CustomLM(model="adapter/reasoning", provider=AdapterProvider(), cache=False)
    program = dspy.ChainOfThought("question -> answer")

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = program(question="Why use typed requests?")

    assert result.reasoning == "typed reasoning"
    assert result.answer == "typed answer"


def test_custom_lm_runs_through_json_adapter() -> None:
    lm = CustomLM(model="adapter/json", provider=AdapterProvider(), cache=False)
    predict = dspy.Predict("question -> answer")

    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        result = predict(question="Return structured output")

    assert result.answer == "typed answer"
    assert lm.supports_response_schema is False
    assert lm.supported_params == {"response_format"}
