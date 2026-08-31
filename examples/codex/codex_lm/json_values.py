"""Narrowing helpers for values parsed from wire JSON."""

from __future__ import annotations

from typing import Any, cast


def as_json_dict(value: object) -> dict[str, Any] | None:
    """Narrow parsed JSON to a string-keyed dict; JSON object keys are strings."""
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return None


def get_dict(mapping: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    """Return ``mapping[key]`` when it is a JSON object, else ``None``."""
    if mapping is None:
        return None
    return as_json_dict(mapping.get(key))


def get_list(mapping: dict[str, Any] | None, key: str) -> list[Any] | None:
    """Return ``mapping[key]`` when it is a JSON array, else ``None``."""
    if mapping is None:
        return None
    value: object = mapping.get(key)
    if isinstance(value, list):
        return cast("list[Any]", value)
    return None


def get_str(mapping: dict[str, Any] | None, key: str) -> str | None:
    """Return ``mapping[key]`` when it is a string, else ``None``."""
    if mapping is None:
        return None
    value: object = mapping.get(key)
    if isinstance(value, str):
        return value
    return None


def get_int(mapping: dict[str, Any] | None, key: str) -> int | None:
    """Return ``mapping[key]`` when it is an integer, else ``None``."""
    if mapping is None:
        return None
    value: object = mapping.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
