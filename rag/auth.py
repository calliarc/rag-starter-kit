"""Collection-level access control: each API key maps to the collections it may read and write."""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


@dataclass(frozen=True)
class Principal:
    key_id: str
    collections: frozenset[str]

    @property
    def is_admin(self) -> bool:
        return "*" in self.collections

    def can_access(self, collection: str) -> bool:
        return self.is_admin or collection in self.collections


def resolve_principal(api_keys: dict[str, list[str]], presented: str | None) -> Principal | None:
    """Return the principal for ``presented``, using constant-time comparison. ``None`` if invalid."""
    if not api_keys:
        return Principal(key_id="anonymous", collections=frozenset({"*"}))
    if not presented:
        return None
    match: Principal | None = None
    for i, (key, collections) in enumerate(api_keys.items()):
        if hmac.compare_digest(key.encode(), presented.encode()):
            match = Principal(key_id=f"key-{i}", collections=frozenset(collections))
    return match


def get_principal(request: Request, api_key: str | None = Depends(_api_key_header)) -> Principal:
    settings = request.app.state.settings
    principal = resolve_principal(settings.api_keys, api_key)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
    return principal


def require_collection(principal: Principal, collection: str) -> None:
    if not principal.can_access(collection):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="access to collection denied")
