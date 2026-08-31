"""Run DSPy programs on a ChatGPT/Codex subscription over the Responses API.

The package is split by domain so each file stays small and readable:

- ``auth``: read, refresh, and persist the Codex CLI's credentials.
- ``translate``: normalized DSPy request to Responses API request body.
- ``transport``: streamed HTTP calls to the Responses endpoint.
- ``sse``: Server-Sent Events parsing.
- ``response``: final Responses object to typed DSPy response.
- ``errors``: backend failures to typed ``dspy.LMError`` types.
- ``provider`` and ``lm``: the DSPy-facing ``CodexProvider`` and ``CodexLM``.
"""

from .lm import CodexLM
from .provider import CodexProvider

__all__ = ["CodexLM", "CodexProvider"]
