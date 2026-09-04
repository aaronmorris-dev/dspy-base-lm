# Changelog

## v0.1.0 (2026-08-31)

### 🚀 Enhancements

- **CustomLM**: typed `LMRequest -> LMResponse` LM with a provider interface, `dspy.cache` memoization, provider error normalization, and reconstruction-safe state
- **LMProvider**: single extension point with capability declarations and sync/async completion
- **response_format**: accept finite JSON-like schema configuration without persisting runtime objects
- **examples/codex**: complete ChatGPT/Codex-subscription provider over `codex exec`

### 🏡 Chore

- adopt uv_build, poethepoet tasks, curated Ruff ruleset, and strict basedpyright
- add MIT license, changelog, and release workflow

### Contributors

- Aaron Morris <35127085+aaronmorris-dev@users.noreply.github.com>
