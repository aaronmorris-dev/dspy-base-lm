from __future__ import annotations

import dspy

from dspy_base_lm import CustomLM, LMProvider


class CountingProvider(LMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        self.calls += 1
        return dspy.LMResponse.from_text(f"response-{self.calls}", model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


def test_cache_identity_covers_the_behavioral_request_fields() -> None:
    # Given requests that independently vary model, messages, tools, and extensions
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=True)
    provider = CountingProvider()
    lm = CustomLM(model="identity/default", provider=provider)
    tool = dspy.core.LMToolSpec(name="lookup", parameters={"type": "object"})
    requests = [
        dspy.LMRequest.from_call(model="identity/model-a", prompt="same"),
        dspy.LMRequest.from_call(model="identity/model-b", prompt="same"),
        dspy.LMRequest.from_call(model="identity/model-a", prompt="different"),
        dspy.LMRequest.from_call(model="identity/model-a", prompt="same", tools=[tool]),
        dspy.LMRequest.from_call(
            model="identity/model-a",
            prompt="same",
            extensions={"region": "east"},
        ),
    ]

    # When every request is completed and then repeated
    first_responses = [lm.forward(request) for request in requests]
    cached_responses = [lm.forward(request) for request in requests]

    # Then each behavioral shape owns one entry and every exact repeat is a hit
    assert provider.calls == len(requests)
    assert [response.text for response in first_responses] == [
        f"response-{index}" for index in range(1, len(requests) + 1)
    ]
    assert all(response.cache_hit for response in cached_responses)
