from __future__ import annotations

from typing import Any

import dspy
import pytest

from dspy_base_lm import CustomLM, LMProvider


class LifecycleProvider(LMProvider):
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        return dspy.LMResponse.from_text("lifecycle", model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class ReconstructableLM(CustomLM):
    def infer_provider(self) -> LMProvider:
        return LifecycleProvider()


def test_copy_shares_runtime_provider_but_isolates_dspy_state() -> None:
    provider = LifecycleProvider()
    lm = CustomLM(model="state/copy", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="record history")
    lm(request)

    copied = lm.copy(temperature=0.7)

    assert copied.provider is provider
    assert copied.history == []
    assert len(lm.history) == 1
    assert copied.kwargs["temperature"] == 0.7


def test_inferred_provider_state_reconstructs_without_runtime_state() -> None:
    lm = ReconstructableLM(model="test/reconstruct", cache=False)

    state = lm.dump_state()
    restored = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)

    assert "provider" not in state
    assert isinstance(restored, ReconstructableLM)
    assert restored.model == "test/reconstruct"


def test_provider_replacement_makes_a_reconstructable_copy_runtime_only() -> None:
    lm = ReconstructableLM(model="test/replaced-provider")
    replacement = LifecycleProvider()

    copied = lm.copy(provider=replacement)

    assert copied.provider is replacement
    with pytest.raises(dspy.LMConfigurationError, match="runtime-only"):
        copied.dump_state()


def test_injected_provider_state_fails_before_runtime_state_can_be_saved() -> None:
    lm = CustomLM(model="state/injected", provider=LifecycleProvider())

    with pytest.raises(dspy.LMConfigurationError, match="runtime-only"):
        lm.dump_state()


def test_runtime_objects_are_rejected_under_innocuous_extension_keys() -> None:
    runtime_value = object()

    with pytest.raises(dspy.LMConfigurationError, match="JSON-like"):
        CustomLM(
            model="state/runtime-value-boundary",
            provider=LifecycleProvider(),
            extensions={"backend": runtime_value},
        )


def test_native_dspy_config_types_are_normalized_to_safe_persistent_data() -> None:
    reasoning = dspy.core.LMReasoningConfig(effort="high")

    lm = CustomLM(
        model="state/native-config",
        provider=LifecycleProvider(),
        reasoning=reasoning,
    )

    assert lm.kwargs["reasoning"] == {
        "effort": "high",
        "max_tokens": None,
        "summary": None,
    }


def test_runtime_objects_are_rejected_in_known_persistent_config() -> None:
    runtime_value = object()

    with pytest.raises(dspy.LMConfigurationError, match="response_format"):
        CustomLM(
            model="state/known-value-boundary",
            provider=LifecycleProvider(),
            response_format=runtime_value,
        )


def test_json_schema_response_format_is_safe_persistent_config() -> None:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }

    lm = CustomLM(
        model="state/schema",
        provider=LifecycleProvider(),
        response_format=response_format,
    )

    assert lm.kwargs["response_format"] == response_format


def test_cyclic_persistent_extensions_raise_a_typed_configuration_error() -> None:
    extensions: dict[str, Any] = {}
    extensions["nested"] = extensions

    with pytest.raises(dspy.LMConfigurationError, match="JSON-like"):
        CustomLM(
            model="state/cyclic-extension",
            provider=LifecycleProvider(),
            extensions=extensions,
        )


def test_arbitrary_json_extension_keys_are_preserved() -> None:
    extensions = {"native": {"authorization": "custom-scheme"}}

    lm = CustomLM(
        model="state/extension-boundary",
        provider=LifecycleProvider(),
        extensions=extensions,
    )

    assert lm.kwargs["extensions"] == extensions
