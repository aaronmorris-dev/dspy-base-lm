# Example: DSPy on a ChatGPT/Codex subscription

A completed `CustomLM` integration that runs DSPy programs against a ChatGPT
subscription through the [Codex CLI](https://developers.openai.com/codex/cli/).
No API key is needed: the Codex CLI holds the subscription credentials, and the
provider never touches them.

## How it works

- [`codex_lm.py`](codex_lm.py) defines `CodexProvider`, a subprocess
  `LMProvider` that runs each request as one ephemeral, read-only-sandboxed
  `codex exec --json` turn, and `CodexLM`, the reconstructable `CustomLM`
  subclass (`infer_provider()` works because authentication is ambient).
- The provider owns the whole transport: argv construction, prompt framing
  over stdin, JSONL event parsing, usage capture, backend error mapping,
  timeout, and retries for transient failures.
- [`demo.py`](demo.py) runs `dspy.Predict` end to end and prints the answer
  plus token usage.

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

| DSPy | Codex CLI |
| --- | --- |
| `model` (`codex/<name>`) | `--model <name>`; unprefixed models are rejected |
| `messages` (text only) | one prompt with `[role]` sections on stdin |
| `config.reasoning.effort` | `-c model_reasoning_effort=...` |
| `config.response_format` | JSON schema via `--output-schema`; `json_object` relies on the adapter prompt |
| usage on `turn.completed` | `LMUsage` (input, output, reasoning, cache read/write) |
| `error` / `turn.failed` events | most specific `dspy.LMError` (auth, rate limit, context window, ...) |

Everything else (`temperature`, `top_p`, `stop`, `n>1`, `logprobs`, tools,
`prompt_cache`, extensions) is rejected with `LMUnsupportedFeatureError`
rather than silently dropped — `codex exec` has no controls for them.

## Limitations

- Codex is an agent, not a raw completion endpoint: it may run read-only shell
  commands while producing an answer, and every turn carries Codex's own
  system prompt (expect ~16k input tokens of overhead per uncached call).
- One candidate per request (`n=1`), no caller-defined tools, no logprobs, and
  text-only inputs.
- Backend error mapping is best-effort string/status matching on the CLI's
  failure events.
