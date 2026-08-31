"""Run a small DSPy program on a ChatGPT/Codex subscription.

Usage: uv run examples/codex/demo.py
"""

from typing import TYPE_CHECKING, cast

import dspy

from codex_lm import CodexLM

if TYPE_CHECKING:
    from collections.abc import Mapping


def main() -> None:
    """Answer one question through DSPy's Predict module via the Codex CLI."""
    lm = CodexLM(model="codex/gpt-5.6-sol")
    dspy.configure(lm=lm)

    predict = dspy.Predict("question -> answer")
    result = predict(question="In one sentence, what is DSPy?")

    answer = cast("str", result.answer)
    # History entries are typed records that read like mappings.
    entry = cast("Mapping[str, object]", lm.history[-1])
    print(f"answer: {answer}")
    print(f"usage: {entry['usage']}")


if __name__ == "__main__":
    main()
