"""Persistent configuration, copy semantics, and reconstruction state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy
import pytest

from dspy_base_lm import CustomLM, LMProvider

if TYPE_CHECKING:
    from conftest import CapturingProvider


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


def test_rollout_id_is_stored_flat_and_grouped_per_request(
    capturing_provider: CapturingProvider,
) -> None:
    # Given a rollout identifier passed at construction, as dspy.LM accepts
    lm = CustomLM(
        model="state/rollout",
        provider=capturing_provider,
        cache=False,
        rollout_id=7,
    )

    # When a public call flows through DSPy's request normalization
    _ = lm("group my configuration")

    # Then storage stays flat like BaseLM and grouping happens per request
    assert lm.kwargs["rollout_id"] == 7
    assert capturing_provider.last_request is not None
    assert capturing_provider.last_request.config.cache is not None
    assert capturing_provider.last_request.config.cache.rollout_id == 7


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


def test_runtime_only_providers_refuse_to_serialize() -> None:
    # Given an injected provider and a copy that replaces a reconstructable one
    injected = CustomLM(model="state/injected", provider=LifecycleProvider())
    reconstructable = ReconstructableLM(model="state/replaced-provider")
    replacement = LifecycleProvider()
    replaced = reconstructable.copy(provider=replacement)

    # When reconstruction state is requested
    # Then both runtime-only LMs fail before producing unusable state
    with pytest.raises(dspy.LMConfigurationError, match="runtime-only"):
        injected.dump_state()
    assert replaced.provider is replacement
    with pytest.raises(dspy.LMConfigurationError, match="runtime-only"):
        replaced.dump_state()


def _cyclic_extensions() -> dict[str, Any]:
    extensions: dict[str, Any] = {}
    extensions["nested"] = extensions
    return extensions


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("extensions", {"backend": object()}),
        ("extensions", _cyclic_extensions()),
        ("response_format", object()),
    ],
    ids=["runtime-extension", "cyclic-extension", "runtime-response-format"],
)
def test_unsafe_persistent_config_is_rejected(field: str, value: object) -> None:
    # Given non-JSON runtime state offered as persistent configuration
    # When the LM is constructed
    # Then validation fails with a typed configuration error
    with pytest.raises(dspy.LMConfigurationError):
        CustomLM(
            model="state/unsafe-config",
            provider=LifecycleProvider(),
            **{field: value},
        )


def test_safe_persistent_config_is_preserved() -> None:
    # Given native DSPy config objects, a JSON schema, and arbitrary JSON extensions
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }
    extensions = {"native": {"authorization": "custom-scheme"}}
    lm = CustomLM(
        model="state/safe-config",
        provider=LifecycleProvider(),
        reasoning=dspy.core.LMReasoningConfig(effort="high"),
        response_format=response_format,
        extensions=extensions,
    )

    # Then values are stored flat as finite reconstruction-safe data
    assert lm.kwargs["reasoning"] == {"effort": "high", "max_tokens": None, "summary": None}
    assert lm.kwargs["response_format"] == response_format
    assert lm.kwargs["extensions"] == extensions
