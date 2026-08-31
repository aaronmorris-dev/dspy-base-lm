<div align="center">

# dspy-base-lm

A typed, provider-neutral foundation for building custom [DSPy](https://github.com/stanfordnlp/dspy) language models.

</div>

## Table of Contents

- [About](#about)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Example: ChatGPT/Codex subscription](#example-chatgptcodex-subscription)
- [Compatibility](#compatibility)
- [Development](#development)
- [License](#license)

## About

DSPy's built-in `dspy.LM` is hardwired to LiteLLM. When your model lives behind an
SDK, a private API, a subprocess, or a local runtime, you need a custom LM — and a
bare `dspy.BaseLM` subclass leaves caching, error discipline, and state safety to you.

`dspy-base-lm` is a second implementation of the same `BaseLM` contract where the only
new idea is a **provider interface**. You implement transport translation; everything
else is inherited from DSPy.

- **`CustomLM`** — a `BaseLM` on DSPy's typed `LMRequest -> LMResponse` contract that
  adds request memoization via `dspy.cache`, error normalization at the provider
  boundary, and reconstruction-safe state. History, usage, callbacks, adapters, and
  copy semantics are native DSPy behavior.
- **`LMProvider`** — the one extension point. Implement `complete` and `acomplete`;
  optionally declare capabilities (`supports_function_calling`, `supports_reasoning`,
  `supports_response_schema`, `supported_params`).
- **No parallel framework.** No LiteLLM shims, no custom request/response types, no
  transport base classes. Providers own their runtime clients, retries, and mapping of
  backend failures to `dspy.LMError` types.

Requires **Python 3.10+**, **DSPy 3.3**, and **Pydantic 2+**.

## Quick Start

### Install

Install from source with uv (recommended):

```sh
uv add git+https://github.com/aaronmorris-dev/dspy-base-lm
```

Or with pip:

```sh
pip install git+https://github.com/aaronmorris-dev/dspy-base-lm
```

### Basic Usage

```python
import dspy

from dspy_base_lm import CustomLM, LMProvider


class EchoProvider(LMProvider):
    def complete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        text = request.messages[-1].text or ""
        return dspy.LMResponse.from_text(text, model=request.model)

    async def acomplete(self, request: dspy.LMRequest, *, num_retries: int) -> dspy.LMResponse:
        return self.complete(request, num_retries=num_retries)


dspy.configure(lm=CustomLM(model="echo/demo", provider=EchoProvider()))
print(dspy.Predict("question -> answer")(question="Does it work?"))
```

Every DSPy module (`Predict`, `ChainOfThought`, adapters, caching, history, usage
tracking) now runs through your provider.

## Usage

### What the provider owns

| Concern | Owner |
| --- | --- |
| Request/response translation, runtime client, retries | `LMProvider` subclass |
| Mapping known backend failures to `dspy.LMError` types | `LMProvider` subclass |
| Capability declaration (`tools`, `reasoning`, `response_format`) | `LMProvider` subclass |
| Caching, history, usage, callbacks, adapters, copy, save/load | `CustomLM` / DSPy |

Unknown exceptions escaping a provider are chained into `dspy.LMUnexpectedError`;
known `dspy.LMError` values pass through unchanged. Raise
`dspy.ContextWindowExceededError` for prompt-too-long failures so DSPy adapters can
react correctly.

### Inject at runtime, or infer for reconstruction

Inject a provider when your application already owns an authenticated client:

```python
lm = CustomLM(model="acme/model", provider=AcmeProvider(client))
```

Injected providers are runtime-only: `dump_state()` refuses to serialize them rather
than producing state that cannot be reconstructed. When the provider can be rebuilt
from ambient configuration, subclass instead:

```python
class AcmeLM(CustomLM):
    def infer_provider(self) -> LMProvider:
        return AcmeProvider.from_environment()
```

`AcmeLM` round-trips through DSPy program save/load. Never serialize credentials to
make reconstruction convenient — persistent configuration must be finite JSON-like
data, and `CustomLM` enforces this at construction and per request.

### Structured outputs

`request.config.response_format` may carry finite JSON-like schema configuration.
Providers translate that data to the backend; runtime classes and objects stay on the
provider rather than entering saved state or cache keys.

### Known DSPy 3.3 limits

Typed streaming and fine-tuning have no adequate native `BaseLM` contract in DSPy 3.3;
this package does not fake them. `n > 1` support, logprobs, and multimodal parts are
provider capabilities — reject what your backend cannot honor rather than dropping it.

## Example: ChatGPT/Codex subscription

[`examples/codex`](examples/codex) is a complete provider that runs DSPy programs on a
ChatGPT subscription by calling the ChatGPT backend's Responses endpoint directly —
streamed HTTP transport, credential refresh, native function calling and JSON schemas,
reasoning controls, usage capture, backend error mapping, retries, and a
reconstructable `CodexLM`. The endpoint is unofficial; the example's README covers
the tradeoff.

```sh
uv run examples/codex/demo.py
```

## Compatibility

This package targets one DSPy minor at a time (currently `>=3.3,<3.4`) and reuses
native DSPy behavior wherever it exists. When upstream changes, the pin is raised and
the contract tests surface any drift — no version shims, no fallback imports.

## Development

```sh
uv sync
uv run poe check   # lint + typecheck + tests
```

Task runner: `poe test`, `poe lint`, `poe format`, `poe typecheck`, `poe release`.

## License

MIT — see [LICENSE](LICENSE).
