"""Narrowing helpers for values parsed from wire JSON."""

from __future__ import annotations

from typing import Any, cast


def as_json_dict(value: object) -> dict[str, Any] | None:
    """Narrow parsed JSON to a string-keyed dict; JSON object keys are strings."""
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return None


def get_dict(mapping: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return ``mapping[key]`` when it is a JSON object, else ``None``."""
    return as_json_dict(mapping.get(key))


def get_list(mapping: dict[str, Any], key: str) -> list[Any] | None:
    """Return ``mapping[key]`` when it is a JSON array, else ``None``."""
    value: object = mapping.get(key)
    return cast("list[Any]", value) if isinstance(value, list) else None


def get_str(mapping: dict[str, Any], key: str) -> str | None:
    """Return ``mapping[key]`` when it is a string, else ``None``."""
    value: object = mapping.get(key)
    return value if isinstance(value, str) else None


def get_int(mapping: dict[str, Any], key: str) -> int | None:
    """Return ``mapping[key]`` when it is an integer, else ``None``."""
    value: object = mapping.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
