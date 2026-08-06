"""Cache-safety checks for the typed provider boundary."""

import math
import re
import uuid
from typing import TypeGuard

import dspy

RUNTIME_CONFIG_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "client",
        "connection",
        "credential",
        "credentials",
        "handle",
        "password",
        "pat",
        "process",
        "refresh_token",
        "secret",
        "session",
        "subprocess",
        "token",
        "transport",
    }
)

_RUNTIME_CONFIG_TOKENS = frozenset(
    {
        "auth",
        "authorization",
        "client",
        "connection",
        "credential",
        "credentials",
        "handle",
        "password",
        "pat",
        "process",
        "secret",
        "session",
        "subprocess",
        "transport",
    }
)
_TOKEN_CREDENTIAL_PREFIXES = frozenset(
    {"access", "api", "auth", "authorization", "bearer", "credential", "refresh"}
)
_CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_SEPARATOR = re.compile(r"[^a-z0-9]+")


class UncacheableResponseError(Exception):
    """Carry a completed response around DSPy's write-through cache decorator."""

    response: dspy.LMResponse

    def __init__(self, response: dspy.LMResponse) -> None:
        super().__init__()
        self.response = response


def normalize_config_key(key: object) -> str:
    """Normalize a configuration key before applying runtime-state policy."""
    separated = _CAMEL_CASE_BOUNDARY.sub("_", str(key).strip())
    return _KEY_SEPARATOR.sub("_", separated.lower()).strip("_")


def is_runtime_config_key(key: object) -> bool:
    """Whether a key names state that belongs on an LMProvider."""
    normalized = normalize_config_key(key)
    if normalized in RUNTIME_CONFIG_KEYS:
        return True

    tokens = frozenset(normalized.split("_"))
    if tokens & _RUNTIME_CONFIG_TOKENS:
        return True
    if "key" in tokens or normalized.endswith(("apikey", "privatekey", "signingkey", "sshkey")):
        return True
    if normalized.endswith("token"):
        return True
    return "token" in tokens and bool(tokens & _TOKEN_CREDENTIAL_PREFIXES)


def provider_cache_partition(provider: object) -> str:
    """Return an opaque runtime partition shared by one provider instance."""
    attribute = "_dspy_base_lm_cache_partition"
    partition = getattr(provider, attribute, None)
    if isinstance(partition, str):
        return partition
    partition = uuid.uuid4().hex
    setattr(provider, attribute, partition)
    return partition


def find_runtime_config_path(value: object, *, prefix: str = "") -> str | None:
    """Return the first path that is unsafe as persistent extension data."""
    return _find_runtime_config_path(value, prefix=prefix, active_containers=set())


def _find_runtime_config_path(
    value: object,
    *,
    prefix: str,
    active_containers: set[int],
) -> str | None:
    if _is_string_object_dict(value):
        return _find_in_mapping(value, prefix=prefix, active_containers=active_containers)
    if _is_object_list(value):
        return _find_in_list(value, prefix=prefix, active_containers=active_containers)
    return None if _is_safe_scalar(value) else prefix or "configuration"


def _find_in_mapping(
    value: dict[str, object],
    *,
    prefix: str,
    active_containers: set[int],
) -> str | None:
    container_id = id(value)
    if container_id in active_containers:
        return prefix or "configuration"
    active_containers.add(container_id)
    try:
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if is_runtime_config_key(key):
                return path
            nested = _find_runtime_config_path(
                item,
                prefix=path,
                active_containers=active_containers,
            )
            if nested is not None:
                return nested
    finally:
        active_containers.remove(container_id)
    return None


def _find_in_list(
    value: list[object],
    *,
    prefix: str,
    active_containers: set[int],
) -> str | None:
    container_id = id(value)
    if container_id in active_containers:
        return prefix or "configuration"
    active_containers.add(container_id)
    try:
        for index, item in enumerate(value):
            nested = _find_runtime_config_path(
                item,
                prefix=f"{prefix}[{index}]",
                active_containers=active_containers,
            )
            if nested is not None:
                return nested
    finally:
        active_containers.remove(container_id)
    return None


def response_is_cache_safe(response: dspy.LMResponse) -> bool:
    """Return whether a completed response contains only safe cache values."""
    try:
        normalized: object = response.model_dump(mode="json")
    except Exception:  # noqa: BLE001 - any serialization failure makes caching unsafe
        return False

    if not _is_safe_value(normalized):
        return False
    if not _is_safe_value(response.provider_response):
        return False
    return all(_is_safe_value(output.provider_output) for output in response.outputs)


def _is_safe_value(value: object, *, active_containers: set[int] | None = None) -> bool:
    if _is_safe_scalar(value):
        return True
    active: set[int] = active_containers if active_containers is not None else set()
    if _is_object_list(value):
        return _is_safe_list(value, active_containers=active)
    if _is_string_object_dict(value):
        return _is_safe_mapping(value, active_containers=active)
    return False


def _is_safe_list(value: list[object], *, active_containers: set[int]) -> bool:
    container_id = id(value)
    if container_id in active_containers:
        return False
    active_containers.add(container_id)
    try:
        return all(_is_safe_value(item, active_containers=active_containers) for item in value)
    finally:
        active_containers.remove(container_id)


def _is_safe_mapping(value: dict[str, object], *, active_containers: set[int]) -> bool:
    container_id = id(value)
    if container_id in active_containers:
        return False
    active_containers.add(container_id)
    try:
        return all(
            not is_runtime_config_key(key)
            and _is_safe_value(item, active_containers=active_containers)
            for key, item in value.items()
        )
    finally:
        active_containers.remove(container_id)


def _is_safe_scalar(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_string_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)
