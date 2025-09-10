from __future__ import annotations
from fastapi import Request

def get_current_user_id(request: Request) -> str | None:
    """
    Return a stable per-user id.
    DEV fallback: read X-User-Id header or default to a fixed demo id.
    Replace with your real session/user lookup.
    """
    uid = request.headers.get("X-User-Id")
    if uid:
        return uid
    # DEV ONLY: single-user mode
    return "demo-user"
