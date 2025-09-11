from __future__ import annotations
import json, sqlite3, threading
from typing import Optional, Dict, Any

_DB_PATH = "tokens.db"
_lock = threading.Lock()

_con = sqlite3.connect(_DB_PATH, check_same_thread=False)
_con.execute("""
CREATE TABLE IF NOT EXISTS google_tokens (
  user_id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  email TEXT,
  scopes TEXT,
  access_token TEXT,
  refresh_token TEXT,
  expiry INTEGER
)
""")
_con.commit()

def _row_to_token(row) -> Dict[str, Any]:
    email, scopes, tok, rtok, exp = row
    return {
        "email": email,
        "scopes": json.loads(scopes or "[]"),
        "token": tok,
        "refresh_token": rtok,
        "expiry": exp,
    }

class TokenStore:
    """SQLite-backed per-user token store (thread-safe)."""

    @classmethod
    def get(cls, user_id: str) -> Optional[Dict[str, Any]]:
        with _lock:
            row = _con.execute(
                "SELECT email, scopes, access_token, refresh_token, expiry "
                "FROM google_tokens WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return _row_to_token(row) if row else None

    @classmethod
    def save(cls, user_id: str, token: Dict[str, Any]) -> None:
        with _lock:
            _con.execute(
                """
                INSERT INTO google_tokens(user_id, provider, email, scopes, access_token, refresh_token, expiry)
                VALUES(?, 'google', ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                  email=excluded.email,
                  scopes=excluded.scopes,
                  access_token=excluded.access_token,
                  refresh_token=COALESCE(excluded.refresh_token, google_tokens.refresh_token),
                  expiry=excluded.expiry
                """,
                (
                    user_id,
                    token.get("email"),
                    json.dumps(token.get("scopes", [])),
                    token.get("token"),
                    token.get("refresh_token"),
                    token.get("expiry"),
                ),
            )
            _con.commit()

    @classmethod
    def delete(cls, user_id: str) -> None:
        with _lock:
            _con.execute("DELETE FROM google_tokens WHERE user_id=?", (user_id,))
            _con.commit()

    # Handy during dev
    @classmethod
    def all(cls) -> Dict[str, Dict[str, Any]]:
        with _lock:
            rows = _con.execute(
                "SELECT user_id, email, scopes, access_token, refresh_token, expiry FROM google_tokens"
            ).fetchall()
        out = {}
        for uid, email, scopes, tok, rtok, exp in rows:
            out[uid] = {
                "email": email,
                "scopes": json.loads(scopes or "[]"),
                "token": tok,
                "refresh_token": rtok,
                "expiry": exp,
            }
        return out
