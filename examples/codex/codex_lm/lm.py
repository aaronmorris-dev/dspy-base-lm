"""The reconstructable Codex-subscription LM."""

from __future__ import annotations

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
