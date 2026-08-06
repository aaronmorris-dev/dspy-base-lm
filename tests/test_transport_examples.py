from __future__ import annotations

import dspy

from dspy_base_lm import CustomLM, LMProvider


class SDKProvider(LMProvider):
    """Represent an SDK client owned entirely by one provider."""

    def __init__(self) -> None:
        super().__init__()
        self.client_name = "fake-sdk-client"

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        return dspy.LMResponse.from_text(self.client_name, model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class HTTPProvider(LMProvider):
    """Represent a provider that owns URL and wire-format translation."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint = "https://model.invalid/v1/generate"

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        return dspy.LMResponse.from_text(self.endpoint, model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class SubprocessProvider(LMProvider):
    """Represent a provider that owns a CLI command and process lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self.command = ("model-cli", "generate")

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        return dspy.LMResponse.from_text(" ".join(self.command), model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


class LocalRuntimeProvider(LMProvider):
    """Represent an in-process model runtime owned by one provider."""

    def __init__(self) -> None:
        super().__init__()
        self.runtime_name = "local-runtime"

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        return dspy.LMResponse.from_text(self.runtime_name, model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


def test_materially_different_transports_require_no_custom_lm_changes() -> None:
    # Given four providers that own materially different runtime concerns
    providers: tuple[LMProvider, ...] = (
        SDKProvider(),
        HTTPProvider(),
        SubprocessProvider(),
        LocalRuntimeProvider(),
    )

    # When each provider is injected through the same typed CustomLM boundary
    responses = []
    states = []
    for index, provider in enumerate(providers):
        lm = CustomLM(model=f"transport/{index}", provider=provider, cache=False)
        request = dspy.LMRequest.from_call(model=f"transport/{index}", prompt="run")
        responses.append(lm.forward(request))
        states.append(lm.dump_state())

    # Then all complete without transport logic or runtime state entering CustomLM
    assert [response.model for response in responses] == [
        "transport/0",
        "transport/1",
        "transport/2",
        "transport/3",
    ]
    assert all("provider" not in state for state in states)
