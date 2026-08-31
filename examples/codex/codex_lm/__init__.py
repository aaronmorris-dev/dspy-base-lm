"""Run DSPy programs on a ChatGPT/Codex subscription over the Responses API."""

from typing_extensions import override

from dspy_base_lm import CustomLM, LMProvider

from .provider import CodexProvider


class CodexLM(CustomLM):
    """A reconstructable ChatGPT/Codex-subscription LM.

    Credentials are ambient (``codex login``), so the provider is inferable
    from the environment and ``dump_state()`` stays free of secrets.
    """

    @override
    def infer_provider(self) -> LMProvider:
        """Reconstruct the provider from the Codex CLI's ambient login."""
        return CodexProvider()


__all__ = ["CodexLM", "CodexProvider"]
