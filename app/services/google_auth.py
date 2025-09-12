from typing import Dict, Any
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
import os, logging

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/calendar.events",
]

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

def _client_config(redirect_uri: str) -> Dict[str, Any]:
    return {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uris": [redirect_uri],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

def start_oauth(state: str, redirect_uri: str) -> str:
    flow = Flow.from_client_config(_client_config(redirect_uri),
                                   scopes=SCOPES, redirect_uri=redirect_uri)
    # IMPORTANT: pass *lowercase string* 'true' (NOT Python True)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",   # <-- lowercase string
        prompt="consent",
        state=state,
    )
    # sanity check + log
    if "include_granted_scopes=True" in auth_url:
        # last-resort guard if some layer uppercases it
        auth_url = auth_url.replace("include_granted_scopes=True", "include_granted_scopes=true")
    logging.info("OAuth auth_url = %s", auth_url)
    return auth_url

def exchange_code_for_tokens(code: str, redirect_uri: str) -> Credentials:
    flow = Flow.from_client_config(_client_config(redirect_uri),
                                   scopes=SCOPES, redirect_uri=redirect_uri)
    flow.fetch_token(code=code)
    return flow.credentials
