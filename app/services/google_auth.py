# app/services/google_auth.py
from __future__ import annotations
import os
from typing import Dict, Any

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

# Use userinfo.* to match what Google returns
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar.events",
]

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")  # e.g. http://127.0.0.1:8000/auth/google/callback

def _client_config() -> Dict[str, Any]:
    return {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uris": [GOOGLE_REDIRECT_URI],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

def start_oauth(state: str) -> str:
    # IMPORTANT: pass the same redirect_uri here and in fetch_token()
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES, redirect_uri=GOOGLE_REDIRECT_URI)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",   # must be lowercase string
        prompt="consent",                # ensures refresh_token for testing
        state=state,
    )
    return auth_url

def exchange_code_for_tokens(code: str) -> Credentials:
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES, redirect_uri=GOOGLE_REDIRECT_URI)
    flow.fetch_token(code=code)
    return flow.credentials

def creds_from_dict(d: dict) -> Credentials:
    return Credentials(
        token=d["token"],
        refresh_token=d.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=SCOPES,
    )
