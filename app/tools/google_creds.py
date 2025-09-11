from __future__ import annotations
from typing import Tuple, Optional

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from app.services.token_store import TokenStore
from app.services.google_auth import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, SCOPES

def get_user_google_creds(user_id: Optional[str]) -> Tuple[Credentials | None, str | None]:
    selected_uid = user_id
    tok = TokenStore.get(user_id) if user_id else None

    # Fallback: if exactly ONE account is connected, use it (dev-friendly, also fine for small pilots)
    if not tok:
        all_tok = TokenStore.all()
        if len(all_tok) == 1:
            selected_uid = next(iter(all_tok.keys()))
            tok = all_tok[selected_uid]

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

    # Refresh if needed and persist the updated token
    try:
        if not creds.valid and creds.refresh_token:
            creds.refresh(GoogleRequest())
            if selected_uid:
                TokenStore.save(selected_uid, {
                    "token": creds.token,
                    "refresh_token": creds.refresh_token or tok.get("refresh_token"),
                    "scopes": list(creds.scopes or tok.get("scopes", [])),
                    "expiry": int(getattr(creds, "expiry", 0).timestamp()) if getattr(creds, "expiry", None) else tok.get("expiry"),
                    "email": tok.get("email"),
                })
    except Exception:
        return None, "AUTH_REQUIRED"

    return creds, None
