from __future__ import annotations
from typing import Tuple, Optional

from google.oauth2.credentials import Credentials
from app.services.token_store import TokenStore
from app.services.google_auth import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SCOPES

def get_user_google_creds(user_id: Optional[str]) -> Tuple[Credentials | None, str | None]:
    tok = TokenStore.get(user_id) if user_id else None

    # DEV FALLBACK: if exactly one user is connected, use that one
    if not tok:
        try:
            all_tokens = TokenStore.all()  # you'll add this classmethod below
            if len(all_tokens) == 1:
                tok = next(iter(all_tokens.values()))
        except Exception:
            pass

    if not tok:
        return None, "AUTH_REQUIRED"

    creds = Credentials(
        token=tok.get("token"),
        refresh_token=tok.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=SCOPES,
    )
    return creds, None
