# app/api/google_oauth.py
from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse

from app.services.google_auth import start_oauth, exchange_code_for_tokens
from app.services.token_store import TokenStore
from app.services.session import get_current_user_id

router = APIRouter(prefix="/auth/google", tags=["google-auth"])

@router.get("/start")
def google_auth_start(request: Request):
    user_id = get_current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not signed in")
    state = f"user:{user_id}"
    url = start_oauth(state)
    return JSONResponse({"auth_url": url})

@router.get("/callback")
def google_auth_callback(code: str, state: str, request: Request):
    # validate state; extract user_id
    if not state.startswith("user:"):
        raise HTTPException(status_code=400, detail="Bad state")
    user_id = state.split(":", 1)[1]

    try:
        creds = exchange_code_for_tokens(code)
        TokenStore.save(user_id, {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "scopes": list(creds.scopes or []),
            "expiry": None,
        })

        # Success: notify opener and close
        html = """
        <script>
          try { window.opener && window.opener.postMessage({type:"google-auth-complete"},"*"); }
          catch(e) {}
          window.close();
        </script>
        <p>Google sign-in complete. You can close this window.</p>
        """
        return HTMLResponse(html)

    except Exception as e:
        # Failure: send a message the opener can catch (optional), and show the error
        html = f"""
        <script>
          try {{ window.opener && window.opener.postMessage({{type:"google-auth-failed", error: "{str(e).replace('"','\\"')}"}},"*"); }}
          catch(err) {{}}
        </script>
        <pre>OAuth callback failed:\n{str(e)}</pre>
        """
        return HTMLResponse(html, status_code=500)
