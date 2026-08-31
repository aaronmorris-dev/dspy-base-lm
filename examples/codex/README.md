# Example: DSPy on a ChatGPT/Codex subscription

A completed `CustomLM` integration that runs DSPy programs against a ChatGPT
subscription by calling the ChatGPT backend's Responses endpoint directly —
the same endpoint the official Codex client uses. The result is a real
language-model interface: full message-list control, your own system prompt,
native function calling, native JSON schemas, reasoning controls, and honest
token usage on every call.

No API key is needed. `codex login` stores the subscription credentials, and
the provider reads and refreshes them in place.

> **This endpoint is unofficial.** OpenAI's supported programmatic surfaces
> for Codex are the CLI, the SDK, and the app-server — all of which drive the
> Codex *agent*, not the model. Calling the Responses endpoint directly is
> what independent agents (pi, opencode) do to use a ChatGPT subscription as
> a raw model, but OpenAI may change or restrict it at any time.

## Layout

One file per domain, so each stays small and readable on its own:

| File | Domain |
| --- | --- |
| [`codex_lm/auth.py`](codex_lm/auth.py) | Read, refresh, and persist `codex login` credentials |
| [`codex_lm/translate.py`](codex_lm/translate.py) | Normalized DSPy request → Responses API request body |
| [`codex_lm/transport.py`](codex_lm/transport.py) | Streamed HTTP calls, one client pair, token-refresh retry |
| [`codex_lm/sse.py`](codex_lm/sse.py) | Server-Sent Events parsing |
| [`codex_lm/response.py`](codex_lm/response.py) | Final Responses object → typed `dspy.LMResponse` |
| [`codex_lm/errors.py`](codex_lm/errors.py) | Backend failures → typed `dspy.LMError` types |
| [`codex_lm/provider.py`](codex_lm/provider.py) | `CodexProvider`: capabilities, retries, orchestration |
| [`codex_lm/lm.py`](codex_lm/lm.py) | `CodexLM`: reconstructable `CustomLM` subclass |

`CodexLM.infer_provider()` works because authentication is ambient: the
provider carries no secrets, so `dump_state()` stays safe and saved programs
reload anywhere `codex login` has run.

## Prerequisites

1. Install the Codex CLI and sign in with your ChatGPT account: `codex login`.
2. From the repository root: `uv sync`.

## Run

```sh
uv run examples/codex/demo.py
```

Each call consumes your ChatGPT subscription's Codex usage. DSPy's cache is on
by default, so repeated identical requests are served locally.

## Mapping decisions

| DSPy | Responses endpoint |
| --- | --- |
| `model` (`codex/<name>`) | `model: <name>`; unprefixed models are rejected |
| leading system messages | `instructions` (later system messages become `developer` items) |
| `messages` (text and images) | role-tagged `input` items |
| `tools` / `config.tool_choice` | native function tools and tool choice |
| `config.response_format` | `text.format`: Pydantic classes, JSON schemas, and `json_object` |
| `config.reasoning.effort` / `.summary` | `reasoning: {effort, summary}`; summaries come back as thinking parts |
| `config.prompt_cache.key` | `prompt_cache_key` (provider-side prefix caching) |
| `usage` on `response.completed` | `LMUsage` (input, output, reasoning, cache read/write) |
| HTTP errors and `response.failed` | most specific `dspy.LMError` (auth, rate limit, billing, context window, ...) |

The endpoint fixes its own sampling controls, so `temperature`, `top_p`,
`max_tokens`, `stop`, `n>1`, `logprobs`, and extensions are rejected with
`LMUnsupportedFeatureError` rather than silently dropped — verified against
the live backend, which refuses them.

## Authentication

- Credentials are read from `$CODEX_HOME/auth.json` (default `~/.codex`),
  written by `codex login`.
- When the backend rejects a token, the provider runs one OAuth
  refresh-token exchange against the Codex CLI's own client id and writes the
  new tokens back in the CLI's format, so `codex` itself keeps working.
- Credentials are never serialized into DSPy state.

## Limitations

- One candidate per request (`n=1`), no logprobs, no sampling controls.
- Reasoning content is returned as summaries; encrypted reasoning is not
  replayed on later turns.
- Backend error mapping is best-effort status/code/message matching.
- The endpoint is unofficial and can change without notice; if it breaks,
  check what pi and opencode currently send.
