# Workflow: turn an unknown model into a custom DSPy LM

This is a decision-and-evidence workflow, not a requirement to write a custom LM. Start
with the model's native interface, prove what it can do, and introduce `CustomLM` only
when ordinary `dspy.LM` cannot reach it. The result is one provider that translates a
transport into DSPy's `LMRequest -> LMResponse` contract.

The package intentionally has two layers:

- `CustomLM` owns the stable DSPy boundary: provider selection, DSPy request caching,
  dispatch, error containment, capability delegation, and supported lifecycle methods.
- `LMProvider` implementations own volatile details: SDK or wire calls, request and
  response translation, retries, known backend errors, credentials, and runtime clients.

There is no transport abstraction beneath `LMProvider`. An SDK, HTTP endpoint, RPC
client, subprocess, or in-process runtime is simply private provider implementation.

## What you will produce

By the end, keep these five artifacts beside your provider:

1. A workload brief explaining why this model is being considered.
2. A native capability profile backed by small probes.
3. Request, response, error, and retry translation maps.
4. An `LMProvider` plus either an injected `CustomLM` or reconstructable subclass.
5. A validation ledger showing which DSPy features actually passed.

Do not declare a capability from marketing copy or a similarly named SDK method. A
capability is supported only after a native probe and a DSPy integration scenario pass.

## Gate 0: decide whether a custom LM is necessary

Write the workload brief before choosing a model:

| Question | Your answer |
| --- | --- |
| What must the model produce? | |
| Which inputs are required: text, images, audio, documents, tools? | |
| Which controls matter: temperature, token limit, stop, reasoning, schema? | |
| What latency, privacy, offline, deployment, or cost constraints apply? | |
| Is streaming required, or merely desirable? | |
| Will the project fine-tune or reinforce the model? | |

Then follow this decision sequence:

1. Can current `dspy.LM` already address the model through a supported provider?
2. Does a documented OpenAI-compatible endpoint work without lossy translation?
3. Can configuration alone select the endpoint and model?
4. Does the model require a custom SDK, nonstandard HTTP/RPC protocol, CLI, subscription
   session, browser-controlled process, or local runtime?

Use normal `dspy.LM` when the first three answers are sufficient. Build a provider only
for the fourth case or when required typed features cannot cross the existing adapter.

**Exit evidence:** one written sentence names the unsupported integration seam that
justifies a custom provider.

## Gate 1: establish a native baseline

Call the model through its native interface before importing DSPy. Use the smallest
possible request and record:

- model identifier and provider-qualified identity;
- authentication source without recording its value;
- sync and async clients;
- request and response examples with secrets removed;
- timeout, cancellation, retry, and rate-limit behavior;
- provider request ID and usage fields;
- native exception classes and status codes.

Probe native failures deliberately: missing credentials, unknown model, invalid request,
timeout, rate limit if a safe sandbox supports it, context overflow, and server failure.
This is how the provider later distinguishes known errors from unexpected bugs.

**Exit evidence:** the smallest native success and each reproducible native failure have
sanitized inputs and observed outputs.

## Gate 2: discover capabilities

Create a capability matrix and mark every row `supported`, `unsupported by provider`,
`blocked by DSPy 3.3`, or `not applicable`.

| Capability | Native probe | DSPy scenario | Evidence |
| --- | --- | --- | --- |
| Text generation | Minimal call | `dspy.Predict` | |
| Multiple candidates | Native `n > 1` equivalent | typed direct call | |
| System/developer roles | Role distinction probe | `LMRequest` messages | |
| Function calling | Tool schema and call | adapter/tool flow | |
| Response schema | Invalid-output rejection | `JSONAdapter` | |
| Reasoning | Native control and output | `ChainOfThought` | |
| Image/audio/document input | One input per media type | typed parts | |
| Citations/refusal | Trigger representative output | typed response parts | |
| Usage and cost | Compare native totals | `dspy.track_usage()` | |
| Async | Native concurrent calls | `acall` | |
| Streaming | Native deltas | current DSPy hook | |
| Fine-tuning | Sandbox job | `finetune` | |
| Reinforcement | Sandbox job | current DSPy job | |

Probe controls independently. A parameter is not supported until changing it changes a
request accepted by the backend or produces observably different backend behavior. Also
record limits: maximum tools, schema restrictions, accepted media sources, context size,
candidate limits, and cancellation guarantees.

**Exit evidence:** every capability has a classification, a probe result, and a reason.

## Gate 3: design the typed boundary

### Choose model identity and provider ownership

Use a provider-qualified model identifier such as `acme/model-name`. It is part of DSPy's
cache identity, history, errors, and usage tracking. Avoid generic identifiers such as
`model` that could collide across transports.

The provider may own live SDK clients, authenticated sessions, connections, subprocess
handles, or an in-memory runtime. These objects are runtime-only. Never put them in:

- `LMRequest.metadata` or `LMConfig.extensions`;
- `LMResponse.provider_data` or `metadata`;
- LM constructor kwargs that DSPy saves;
- cache-key material.

Persistent extensions and request metadata must contain finite, JSON-like values only.
`CustomLM` rejects runtime objects, reference cycles, and credential-shaped keys before
provider dispatch, caching, history, or state serialization. This guard is a last line of
defense, not a secret store: keep all authentication material on the runtime provider.

`LMResponse.provider_response`, provider data, metadata, and each output's provider data may
retain safe native values for immediate inspection. If a completed response contains a
runtime object, reference cycle, or credential-shaped key, `CustomLM` returns the live
response but deliberately bypasses DSPy's cache for that result.

### Map requests field by field

| DSPy source | Provider destination | Unsupported behavior |
| --- | --- | --- |
| `request.model` | native model or deployment ID | `LMUnsupportedModelError` |
| `request.messages` | native roles/content | `LMUnsupportedFeatureError` |
| `request.tools` | native tool definitions | unsupported `function_calling` |
| `request.config.temperature` | sampling control | omit or reject explicitly |
| `request.config.max_tokens` | output token control | provider limit error |
| `request.config.response_format` | JSON/schema control | unsupported `response_schema` |
| `request.config.reasoning` | reasoning controls | unsupported `reasoning` |
| `request.config.prompt_cache` | provider-side prompt cache | omit or reject explicitly |
| `request.config.extensions` | documented provider extras | reject unknown keys at boundary |

Do not pass DSPy's memoization setting to the backend. `request.config.cache` controls
whether DSPy skips the provider call. Provider-side prompt caching is the separate
`request.config.prompt_cache` field.

### Map responses without flattening

Always construct `dspy.LMResponse`. Preserve every native field that has a DSPy home:

- one `LMOutput` per candidate;
- text, thinking, tool call, citation, refusal, and applicable media parts;
- finish reason, truncation, and log probabilities;
- `LMUsage`, cost, response ID, safe provider data, and metadata.

Use `LMResponse.from_text(...)` only for truly text-only backends. Do not return strings,
OpenAI-shaped dictionaries, SDK response objects, or legacy completion lists.

### Map known errors in the provider

Map native failures to the most specific DSPy error:

| Native condition | DSPy error |
| --- | --- |
| connection, DNS, TLS, broken pipe | `LMTransportError` |
| missing local configuration | `LMNotConfiguredError` |
| authentication rejected | `LMAuthError` |
| billing or quota rejected | `LMBillingError` |
| rate limited | `LMRateLimitError` |
| invalid request | `LMInvalidRequestError` |
| context too large | `ContextWindowExceededError` |
| unsupported model | `LMUnsupportedModelError` |
| deadline exceeded | `LMTimeoutError` |
| provider 5xx | `LMServerError` |

Preserve model, provider, provider code, status, request ID, and retry-after metadata when
known. Re-raise existing `dspy.LMError` values unchanged. `CustomLM` converts only unknown
exceptions crossing the provider call boundary into a chained `LMUnexpectedError`.

### Assign retries once

`CustomLM` passes `num_retries` to the provider. The provider owns retry classification,
backoff, deadlines, and cleanup. Never add a second retry loop around `CustomLM`; nested
retries multiply calls and obscure the final failure.

**Exit evidence:** the four translation maps are complete, and every dropped field has an
explicit reason.

## Gate 4: implement the smallest vertical slice

Start with text-only sync and async completion. This runnable reference requires no
credentials:

```python
import anyio
import dspy

from dspy_base_lm import EchoLM


lm = EchoLM(model="echo/reference", cache=False)
request = dspy.LMRequest.from_call(model=lm.model, prompt="Hello typed DSPy")

sync_response = lm(request)
assert isinstance(sync_response, dspy.LMResponse)
assert sync_response.text == "Hello typed DSPy"


async def main() -> None:
    async_response = await lm.acall(request)
    assert isinstance(async_response, dspy.LMResponse)
    assert async_response.text == "Hello typed DSPy"


anyio.run(main)
```

Then implement the provider itself:

```python
import dspy

from dspy_base_lm import CustomLM, LMProvider


class MyProvider(LMProvider):
    def __init__(self, client: "MySDKClient") -> None:
        super().__init__()
        self.client = client  # Runtime-only. Never place this in LM state or requests.

    def supported_params(self, model: str) -> frozenset[str]:
        return frozenset({"temperature", "max_tokens"})

    def supports_function_calling(self, model: str) -> bool:
        return model == "my-provider/tool-model"

    def supports_reasoning(self, model: str) -> bool:
        return False

    def supports_response_schema(self, model: str) -> bool:
        return False

    def complete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        native_request = to_native_request(request)
        native_response = self.client.generate(native_request, retries=num_retries)
        return to_lm_response(native_response, model=request.model)

    async def acomplete(
        self,
        request: dspy.LMRequest,
        *,
        num_retries: int,
    ) -> dspy.LMResponse:
        native_request = to_native_request(request)
        native_response = await self.client.agenerate(native_request, retries=num_retries)
        return to_lm_response(native_response, model=request.model)


lm = CustomLM(model="my-provider/my-model", provider=MyProvider(client))
```

`MySDKClient`, `to_native_request`, and `to_lm_response` are deliberately provider-local.
They are not framework abstractions and should describe the actual backend precisely.

### Inject at runtime or infer for reconstruction

Injection is best when an application already owns a client or authenticated session:

```python
lm = CustomLM(model="acme/model", provider=AcmeProvider(client))
```

Injected providers are not serialized. Loading this bare `CustomLM` state correctly fails
until the application injects a new runtime provider.

Subclass only when the provider can be reconstructed from safe ambient configuration:

```python
class AcmeLM(CustomLM):
    def infer_provider(self) -> LMProvider:
        return AcmeProvider.from_environment()


lm = AcmeLM(model="acme/model")
```

Never serialize credentials to make inference convenient. If a subclass adds persistent,
non-secret configuration beyond `BaseLM.__init__`, it must define an explicit safe state
contract rather than assuming DSPy will discover arbitrary attributes.

### Transport-specific ownership

- **SDK:** provider owns the SDK client, native models, pagination or streaming objects,
  and SDK exception mapping.
- **HTTP/RPC:** provider owns endpoint, headers, timeouts, wire schemas, status mapping,
  and connection reuse. Credentials belong in the runtime client, not the request.
- **CLI/subprocess:** provider owns argv construction, stdin/stdout framing, exit-code
  mapping, cancellation, process cleanup, and bounded stderr capture.
- **Local runtime:** provider owns tokenizer/model handles, device placement, locks,
  batching, cancellation boundaries, and resource shutdown.

None of these choices requires a branch or modification in `CustomLM`.

**Exit evidence:** direct sync and async typed calls return `LMResponse`, and no runtime
client appears in `dump_state()`.

## Gate 5: validate through DSPy

Test public behavior, not private DSPy helper layout. Run the same provider through:

1. explicit `LMRequest` calls, including roles, prior responses, tool results, tools, and
   applicable media parts;
2. `dspy.Predict`;
3. `dspy.ChainOfThought` when reasoning is applicable;
4. `ChatAdapter`;
5. `JSONAdapter` when response schemas or JSON output are applicable;
6. callbacks, history, `dspy.track_usage()`, and disabled history;
7. `copy()` and trusted save/load where reconstruction is supported;
8. DSPy cache hits, cache bypass, and distinct rollout IDs;
9. known error pass-through and unknown error chaining;
10. concurrent sync and async calls against the provider's shared runtime;
11. fine-tuning success and failure when the provider declares it supported;
12. async cancellation and provider-owned resource cleanup where applicable.

For cache verification, equivalent typed requests must call the provider once. A cached
response is marked `cache_hit=True`; DSPy clears usage on the hit because no provider call
occurred. Cache entries are partitioned by the active runtime provider instance, including
when `copy(provider=...)` replaces a provider. Failed calls, partial streams, and responses
containing runtime or credential data must never enter the cache. Keep secrets out of
requests; `CustomLM` rejects recognizable credential keys, but cannot infer intent from an
innocuously named string.

DSPy's `copy()` intentionally shares the provider instance. Provider authors therefore own
thread safety, async safety, connection-pool limits, cancellation, and shutdown for that
shared runtime; `CustomLM` must not clone or serialize it.

Maintain a ledger:

| Scenario | Status | Evidence | Limitation |
| --- | --- | --- | --- |
| Direct typed sync/async | | | |
| Predict and ChatAdapter | | | |
| ChainOfThought | | | |
| JSONAdapter | | | |
| Cache and rollout | | | |
| Callback/history/usage | | | |
| Copy and reconstruction | | | |
| Fine-tuning | | | |
| Streaming | | | |
| Reinforcement | | | |

**Exit evidence:** every supported feature has a passing scenario; unsupported and
upstream-blocked features are explicit rather than silently degraded.

## Gate 6: operational handoff

Freeze a provider contract with:

- supported model identifiers and capabilities;
- accepted DSPy config and extension keys;
- request/response examples with secrets removed;
- native-to-DSPy error mapping and retry policy;
- timeout, concurrency, cancellation, and cleanup expectations;
- cache-safety rules for provider response data;
- reconstruction strategy and credential source;
- upstream DSPy limitations that affect this provider.

Run the project gates with uv:

```text
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pytest
uv build
```

Inspect and install both artifacts in clean uv environments. Verify the installed package,
not the source checkout, with the Echo sync and async workflow.

## DSPy 3.3 limitations to keep visible

- Typed custom `BaseLM` inference, normalization, callbacks, history, usage, caching,
  adapters, copy, and trusted reconstruction are supported by this package.
- Typed streaming event types exist, but DSPy 3.3's public `streamify`/`StreamListener`
  integration still follows the built-in completion chunk path. This package does not
  mirror it; custom typed streaming is **blocked by DSPy 3.3**.
- Direct provider fine-tuning is supported through DSPy's `TrainingJob` contract.
  `BootstrapFinetune` per-LM mappings still assume concrete `dspy.LM` keys and may exclude
  custom `BaseLM` instances.
- Provider launch/kill and reinforcement job annotations name concrete `dspy.LM`, not
  `BaseLM`. This package does not add compatibility wrappers; those features are
  **blocked by DSPy 3.3 for this custom LM boundary**.
- `forward_contract = "typed_lm"` is retained only because DSPy 3.3 requires the explicit
  migration marker. Remove it when the supported DSPy minor no longer requires it.

See [the feature matrix](feature-matrix.md) for the authoritative classification.

## Final completion criterion

A provider author can explain why a custom LM is needed, reproduce the model natively,
classify its capabilities, implement one typed provider, run it through applicable DSPy
modules, and identify current upstream limitations without modifying `CustomLM` or copying
DSPy/LiteLLM internals.
