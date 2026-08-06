from __future__ import annotations

from typing import TYPE_CHECKING, Any

import dspy
import pytest

from dspy_base_lm import CustomLM, EchoLM, LMProvider

if TYPE_CHECKING:
    from dspy.clients.utils_finetune import TrainDataFormat


class CompletedTrainingJob(dspy.TrainingJob):
    def status(self) -> str:
        return "complete" if self.done() else "running"


class LifecycleProvider(LMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.finetunable = True
        self.TrainingJob = CompletedTrainingJob
        self.finetune_model: str | None = None
        self.finetune_data: list[dict[str, Any]] | None = None
        self.finetune_format: TrainDataFormat | str | None = None
        self.finetune_kwargs: dict[str, Any] | None = None

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

    def finetune(
        self,
        job: dspy.TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        _ = job
        self.finetune_model = model
        self.finetune_data = train_data
        self.finetune_format = train_data_format
        self.finetune_kwargs = train_kwargs
        return f"{model}-tuned"


class FailingLifecycleProvider(LifecycleProvider):
    def finetune(
        self,
        job: dspy.TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        _ = self, job, model, train_data, train_data_format, train_kwargs
        message = "native training failed"
        raise RuntimeError(message)


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


def test_echo_state_reconstructs_without_serializing_provider() -> None:
    # Given a reconstructable EchoLM
    lm = EchoLM(model="echo/reconstruct", cache=False)

    # When DSPy dumps and loads trusted custom LM state
    state = lm.dump_state()
    restored = dspy.BaseLM.load_state(state, allow_custom_lm_class=True)

    # Then provider runtime is absent from state and inferred on reconstruction
    assert "provider" not in state
    assert isinstance(restored, EchoLM)
    assert restored.model == "echo/reconstruct"


def test_injected_provider_state_requires_fresh_runtime_configuration() -> None:
    # Given a bare CustomLM with an injected runtime provider
    lm = CustomLM(model="state/injected", provider=LifecycleProvider())
    state = lm.dump_state()

    # When trusted reconstruction is attempted without runtime injection
    with pytest.raises(dspy.LMNotConfiguredError):
        dspy.BaseLM.load_state(state, allow_custom_lm_class=True)

    # Then no provider client or credential-bearing runtime entered state
    assert "provider" not in state


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "apiKey",
        "api_token",
        "auth_token",
        "bearer_token",
        "client",
        "connection",
        "openai_api_key",
        "openaiToken",
        "private-key",
        "provider-token",
        "github_pat",
        "sdk_client",
        "signingKey",
        "ssh_key",
        "token",
        "x-api-key",
    ],
)
def test_runtime_provider_state_is_rejected_as_lm_configuration(key: str) -> None:
    # Given provider runtime state misplaced in persistent LM configuration
    runtime_value = "do-not-persist"

    # When CustomLM is constructed
    with pytest.raises(dspy.LMConfigurationError, match="LMProvider"):
        CustomLM(
            model="state/runtime-boundary",
            provider=LifecycleProvider(),
            **{key: runtime_value},
        )


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


def test_credentials_are_rejected_inside_persistent_extensions() -> None:
    # Given a credential misplaced in persistent provider extensions
    extensions = {"authorization": "Bearer do-not-persist"}

    # When CustomLM is constructed
    with pytest.raises(dspy.LMConfigurationError, match="LMProvider"):
        CustomLM(
            model="state/extension-boundary",
            provider=LifecycleProvider(),
            extensions={"native": extensions},
        )


def test_finetune_returns_a_custom_lm() -> None:
    # Given a lifecycle-capable provider
    provider = LifecycleProvider()
    lm = CustomLM(model="lifecycle/model", provider=provider)

    # When fine-tuning runs through the public LM method
    train_data = [{"prompt": "hello"}]
    train_kwargs = {"epochs": 2}
    job = lm.finetune(
        train_data,
        train_data_format=None,
        train_kwargs=train_kwargs,
    )
    result = job.result(timeout=2)

    # Then the provider owns training work and DSPy owns its job contract
    assert isinstance(result, CustomLM)
    assert result.model == "lifecycle/model-tuned"
    assert result.provider is provider
    assert provider.finetune_model == "lifecycle/model"
    assert provider.finetune_data == train_data
    assert provider.finetune_format is None
    assert provider.finetune_kwargs == train_kwargs
    assert job.done()
    assert job.thread is not None
    assert not job.thread.is_alive()


def test_finetune_preserves_dspys_training_job_failure_semantics() -> None:
    # Given a provider whose native training operation fails
    provider = FailingLifecycleProvider()
    lm = CustomLM(model="lifecycle/failing-model", provider=provider)

    # When DSPy's asynchronous training job completes
    job = lm.finetune([{"prompt": "hello"}], train_data_format=None)
    result = job.result(timeout=2)

    # Then the job resolves with the provider error exactly like DSPy's native LM
    assert isinstance(result, RuntimeError)
    assert str(result) == "native training failed"
    assert job.done()
    assert job.thread is not None
    assert not job.thread.is_alive()


def test_unsupported_finetuning_raises_a_dspy_error() -> None:
    # Given the non-training Echo provider
    lm = EchoLM(model="echo/reference")

    # When unavailable training methods are requested
    with pytest.raises(dspy.LMUnsupportedFeatureError) as finetune_error:
        lm.finetune([], train_data_format=None)
    # Then the error identifies the unsupported DSPy feature
    assert finetune_error.value.features == ["finetuning"]
