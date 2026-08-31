"""Verify the installed package imports and completes one typed call.

Runs under pytest and standalone. The publish workflow executes this file
against the built wheel and source distribution to catch packaging mistakes.
"""

import dspy

from dspy_base_lm import CustomLM, LMProvider


class PackagedProvider(LMProvider):
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        _ = num_retries
        return dspy.LMResponse.from_text("ok", model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


def test_installed_package_completes_a_typed_call() -> None:
    lm = CustomLM(model="packaging/model", provider=PackagedProvider(), cache=False)
    request = dspy.LMRequest.from_call(model=lm.model, prompt="ping")
    response = lm.forward(request)
    assert response.text == "ok"


if __name__ == "__main__":
    test_installed_package_completes_a_typed_call()
    print("package verification passed")
