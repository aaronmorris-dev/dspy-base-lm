# Working in the DSPy custom LM reference

- Target the current supported DSPy minor and its typed `LMRequest -> LMResponse` `BaseLM` contract. Do not implement legacy call contracts, LiteLLM compatibility, version shims, aliases, or fallback imports; raise the DSPy minimum when upstream changes.
- Reuse and inherit DSPy’s native public functions, classes, types, and lifecycle behavior unless `BaseLM` lacks required behavior. Prefer inherited `BaseLM` behavior first, then public DSPy APIs and types, and add the smallest package-owned implementation only when no stable public API exists. Mirror a public `dspy.LM` method or use a private DSPy import only when required for the supported minor; isolate it, document why, and cover it with contract tests. Remove package-owned behavior when DSPy provides an adequate native equivalent.
- Keep `CustomLM` thin. Providers own transport, translation, retries, known backend error mapping, capabilities, and runtime clients; do not create a transport framework or parallel DSPy types.
- Use DSPy caching and typed stream events only. Never serialize credentials, clients, or connections, and never fake unsupported streaming or training.
- Use `uv` only. Require Ruff, strict basedpyright, focused public-behavior tests once contracts stabilize, clean artifacts, and verification against the supported DSPy minor.
