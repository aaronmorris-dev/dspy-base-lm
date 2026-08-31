"""Incremental parsing of Server-Sent Events into JSON payloads."""

from __future__ import annotations

import json
from typing import Any

from .json_values import as_json_dict


class ServerSentEventParser:
    """Assemble ``data:`` lines into complete JSON events.

    The Responses stream carries the event type inside each JSON payload, so
    ``event:``, ``id:``, and comment lines are intentionally ignored.
    """

    def __init__(self) -> None:
        self._data_lines: list[str] = []

    def feed(self, line: str) -> dict[str, Any] | None:
        """Consume one stream line; a blank line completes and returns an event."""
        if line == "":
            return self.flush()
        if line.startswith("data:"):
            self._data_lines.append(line[len("data:") :].removeprefix(" "))
        return None

    def flush(self) -> dict[str, Any] | None:
        """Return the event assembled from any buffered data lines."""
        if not self._data_lines:
            return None
        payload = "\n".join(self._data_lines)
        self._data_lines = []
        try:
            parsed: object = json.loads(payload)
        except json.JSONDecodeError:
            return None  # For example the "[DONE]" sentinel some SSE endpoints send.
        return as_json_dict(parsed)
