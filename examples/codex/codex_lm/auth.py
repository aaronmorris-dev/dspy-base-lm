"""Load, refresh, and persist the Codex CLI's ChatGPT-subscription credentials.

Authentication is ambient: `codex login` writes `$CODEX_HOME/auth.json`, and
this module only reads it, refreshes the OAuth tokens when the backend rejects
them, and writes the refreshed tokens back in the same format so the Codex CLI
keeps working. Credentials never leave this process in any other way.
"""

from __future__ import annotations

import base64
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dspy
import httpx

from .errors import PROVIDER_NAME
from .json_values import as_json_dict, get_dict, get_str

_OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # The Codex CLI's public OAuth client.
_OAUTH_SCOPE = "openid profile email"
_JWT_AUTH_CLAIM = "https://api.openai.com/auth"
_REFRESH_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class CodexTokens:
    """The credentials one Responses request needs."""

    access_token: str
    account_id: str


class CodexAuth:
    """Read tokens written by ``codex login`` and refresh them when rejected."""

    def __init__(self, codex_home: str | None = None) -> None:
        """Locate ``auth.json`` under ``codex_home``, ``$CODEX_HOME``, or ``~/.codex``."""
        home = codex_home or os.environ.get("CODEX_HOME") or "~/.codex"
        self._auth_file = Path(home).expanduser() / "auth.json"
        self._lock = threading.Lock()
        self._tokens: CodexTokens | None = None

    def tokens(self) -> CodexTokens:
        """Return the current credentials, reading ``auth.json`` on first use."""
        with self._lock:
            if self._tokens is None:
                self._tokens = self._tokens_from_file(self._read_file())
            return self._tokens

    def refresh(self, stale_access_token: str) -> CodexTokens:
        """Exchange the stored refresh token for new credentials.

        ``stale_access_token`` identifies the credentials that were rejected:
        when a concurrent request already refreshed them, the newer tokens are
        returned without another token exchange.
        """
        with self._lock:
            if self._tokens is not None and self._tokens.access_token != stale_access_token:
                return self._tokens
            file_data = self._read_file()
            tokens_data = get_dict(file_data, "tokens") or {}
            refresh_token = get_str(tokens_data, "refresh_token")
            if refresh_token is None:
                raise self._not_logged_in("has no refresh token")
            payload = self._exchange_refresh_token(refresh_token)
            for key in ("id_token", "access_token", "refresh_token"):
                value = get_str(payload, key)
                if value is not None:
                    tokens_data[key] = value
            file_data["tokens"] = tokens_data
            file_data["last_refresh"] = datetime.now(timezone.utc).isoformat()
            self._write_file(file_data)
            self._tokens = self._tokens_from_data(tokens_data)
            return self._tokens

    def _exchange_refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Run one OAuth refresh-token exchange against the Codex CLI's client."""
        response = httpx.post(
            _OAUTH_TOKEN_URL,
            json={
                "client_id": _OAUTH_CLIENT_ID,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "scope": _OAUTH_SCOPE,
            },
            timeout=_REFRESH_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            message = (
                f"Refreshing the Codex credentials failed with HTTP {response.status_code}; "
                "run `codex login` again."
            )
            raise dspy.LMAuthError(
                message,
                provider=PROVIDER_NAME,
                status=response.status_code,
            )
        payload = as_json_dict(response.json())
        if payload is None or get_str(payload, "access_token") is None:
            message = "The OAuth token endpoint returned no access token; run `codex login`."
            raise dspy.LMAuthError(message, provider=PROVIDER_NAME)
        return payload

    def _read_file(self) -> dict[str, Any]:
        """Read and parse ``auth.json``, or explain how to create it."""
        try:
            raw = self._auth_file.read_text()
        except FileNotFoundError:
            message = (
                f"No Codex CLI credentials at {self._auth_file}. Install the Codex CLI "
                "and sign in with `codex login` (ChatGPT subscription)."
            )
            raise dspy.LMNotConfiguredError(message, provider=PROVIDER_NAME) from None
        data = as_json_dict(json.loads(raw))
        if data is None:
            raise self._not_logged_in("is not a JSON object")
        return data

    def _write_file(self, data: dict[str, Any]) -> None:
        """Persist refreshed tokens in the Codex CLI's own format and permissions."""
        self._auth_file.write_text(json.dumps(data, indent=2) + "\n")
        self._auth_file.chmod(0o600)

    def _tokens_from_file(self, file_data: dict[str, Any]) -> CodexTokens:
        """Extract subscription tokens, rejecting API-key-only logins."""
        tokens_data = get_dict(file_data, "tokens")
        if tokens_data is None:
            if get_str(file_data, "OPENAI_API_KEY"):
                message = (
                    f"{self._auth_file} holds an API key, not ChatGPT-subscription tokens. "
                    "Use dspy.LM('openai/<model>') for API-key access, or run "
                    "`codex login` with a ChatGPT account."
                )
                raise dspy.LMNotConfiguredError(message, provider=PROVIDER_NAME)
            raise self._not_logged_in("has no tokens")
        return self._tokens_from_data(tokens_data)

    def _tokens_from_data(self, tokens_data: dict[str, Any]) -> CodexTokens:
        """Build ``CodexTokens`` from the ``tokens`` object in ``auth.json``."""
        access_token = get_str(tokens_data, "access_token")
        if access_token is None:
            raise self._not_logged_in("has no access token")
        account_id = get_str(tokens_data, "account_id") or _account_id_from_jwt(access_token)
        if account_id is None:
            raise self._not_logged_in("names no ChatGPT account id")
        return CodexTokens(access_token=access_token, account_id=account_id)

    def _not_logged_in(self, problem: str) -> dspy.LMNotConfiguredError:
        message = f"{self._auth_file} {problem}; run `codex login` again."
        return dspy.LMNotConfiguredError(message, provider=PROVIDER_NAME)


def _account_id_from_jwt(token: str) -> str | None:
    """Read the ChatGPT account id claim from an access token, if present."""
    segments = token.split(".")
    if len(segments) != 3:
        return None
    payload_segment = segments[1]
    padded = payload_segment + "=" * (-len(payload_segment) % 4)
    try:
        claims = as_json_dict(json.loads(base64.urlsafe_b64decode(padded)))
    except (ValueError, json.JSONDecodeError):
        return None
    if claims is None:
        return None
    auth_claim = get_dict(claims, _JWT_AUTH_CLAIM)
    return get_str(auth_claim, "chatgpt_account_id") if auth_claim is not None else None
