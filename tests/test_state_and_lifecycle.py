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


def test_rollout_id_is_stored_flat_and_grouped_per_request() -> None:
    # Given a rollout identifier passed at construction, as dspy.LM accepts
    captured: list[dspy.LMRequest] = []

    class CapturingProvider(LifecycleProvider):
        def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
            captured.append(request)
            return super().complete(request, num_retries=num_retries)

    lm = CustomLM(
        model="state/rollout",
        provider=CapturingProvider(),
        cache=False,
        rollout_id=7,
    )

    # When a public call flows through DSPy's request normalization
    _ = lm("group my configuration")

    # Then storage stays flat like BaseLM and grouping happens per request
    assert lm.kwargs["rollout_id"] == 7
    assert captured[0].config.cache is not None
    assert captured[0].config.cache.rollout_id == 7


def test_copy_shares_runtime_provider_but_isolates_dspy_state() -> None:
    # Given an LM with typed history and a runtime provider
    provider = LifecycleProvider()
    lm = CustomLM(model="state/copy", provider=provider, cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="record history")
    lm(request)

    # When DSPy's native copy behavior is used
    copied = lm.copy(temperature=0.7)

    # Then runtime ownership is shared while mutable DSPy state is isolated
    assert copied.provider is provider
    assert copied.history == []
    assert len(lm.history) == 1
    assert copied.kwargs["temperature"] == 0.7


def test_inferred_provider_state_reconstructs_without_runtime_state() -> None:
    # Given a test-local LM that reconstructs its provider
    lm = ReconstructableLM(model="test/reconstruct", cache=False)

    # When DSPy dumps and loads trusted custom LM state
    state = lm.dump_state()
    restored = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)

    # Then provider runtime is absent from state and inferred on reconstruction
    assert "provider" not in state
    assert isinstance(restored, ReconstructableLM)
    assert restored.model == "test/reconstruct"


def test_provider_replacement_makes_a_reconstructable_copy_runtime_only() -> None:
    # Given a reconstructable LM and a replacement runtime provider
    lm = ReconstructableLM(model="test/replaced-provider")
    replacement = LifecycleProvider()

    # When inherited copy behavior installs the runtime provider
    copied = lm.copy(provider=replacement)

    # Then the copy uses that provider but cannot serialize misleading reconstruction state
    assert copied.provider is replacement
    with pytest.raises(dspy.LMConfigurationError, match="runtime-only"):
        copied.dump_state()


def test_injected_provider_state_fails_before_runtime_state_can_be_saved() -> None:
    # Given a bare CustomLM with an injected runtime provider
    lm = CustomLM(model="state/injected", provider=LifecycleProvider())

    # When reconstruction state is requested without a reconstructable provider
    with pytest.raises(dspy.LMConfigurationError, match="runtime-only"):
        lm.dump_state()


def test_runtime_objects_are_rejected_under_innocuous_extension_keys() -> None:
    # Given a runtime client hidden behind a non-sensitive extension key
    runtime_value = object()

    # When persistent LM configuration is constructed
    with pytest.raises(dspy.LMConfigurationError, match="JSON-like"):
        CustomLM(
            model="state/runtime-value-boundary",
            provider=LifecycleProvider(),
            extensions={"backend": runtime_value},
        )


def test_native_dspy_config_types_are_normalized_to_safe_persistent_data() -> None:
    # Given a native DSPy reasoning configuration
    reasoning = dspy.core.LMReasoningConfig(effort="high")

    # When CustomLM normalizes persistent configuration
    lm = CustomLM(
        model="state/native-config",
        provider=LifecycleProvider(),
        reasoning=reasoning,
    )

    # Then BaseLM retains only its finite reconstruction-safe representation
    assert lm.kwargs["reasoning"] == {
        "effort": "high",
        "max_tokens": None,
        "summary": None,
    }


def test_runtime_objects_are_rejected_in_known_persistent_config() -> None:
    # Given runtime state in DSPy's permissive response-format field
    runtime_value = object()

    # When persistent LM configuration is constructed
    with pytest.raises(dspy.LMConfigurationError, match="response_format"):
        CustomLM(
            model="state/known-value-boundary",
            provider=LifecycleProvider(),
            response_format=runtime_value,
        )


def test_json_schema_response_format_is_safe_persistent_config() -> None:
    # Given a finite JSON-schema-like response format
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }

    # When persistent LM configuration is constructed
    lm = CustomLM(
        model="state/schema",
        provider=LifecycleProvider(),
        response_format=response_format,
    )

    # Then the schema remains available to DSPy's normalized request path
    assert lm.kwargs["response_format"] == response_format


def test_cyclic_persistent_extensions_raise_a_typed_configuration_error() -> None:
    # Given persistent extension data containing a reference cycle
    extensions: dict[str, Any] = {}
    extensions["nested"] = extensions

    # When CustomLM validates reconstruction state
    with pytest.raises(dspy.LMConfigurationError, match="JSON-like"):
        CustomLM(
            model="state/cyclic-extension",
            provider=LifecycleProvider(),
            extensions=extensions,
        )


def test_arbitrary_json_extension_keys_are_preserved() -> None:
    # Given valid JSON configuration with a domain-specific authorization field
    extensions = {"native": {"authorization": "custom-scheme"}}

    # When CustomLM normalizes persistent configuration
    lm = CustomLM(
        model="state/extension-boundary",
        provider=LifecycleProvider(),
        extensions=extensions,
    )

    # Then it preserves the value without guessing intent from its key name
    assert lm.kwargs["extensions"] == extensions
