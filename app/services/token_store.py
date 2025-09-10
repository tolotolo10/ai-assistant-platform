# app/services/token_store.py

from __future__ import annotations
from typing import Optional, Dict, Any

class TokenStore:
    """
    Minimal per-user token store.
    Replace _mem with your DB/Redis for production.
    """
    _mem: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get(cls, user_id: str) -> Optional[Dict[str, Any]]:
        return cls._mem.get(user_id)

    @classmethod
    def save(cls, user_id: str, token: Dict[str, Any]) -> None:
        cls._mem[user_id] = token

    @classmethod
    def delete(cls, user_id: str) -> None:
        cls._mem.pop(user_id, None)
