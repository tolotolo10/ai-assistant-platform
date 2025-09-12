from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from googleapiclient.discovery import build
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
    redirect_uri = str(request.url_for("google_auth_callback"))
    url = start_oauth(state, redirect_uri)
    return JSONResponse({"auth_url": url})

@router.get("/callback", name="google_auth_callback")
def google_auth_callback(code: str, state: str, request: Request):
    if not state.startswith("user:"):
        raise HTTPException(status_code=400, detail="Bad state")
    user_id = state.split(":", 1)[1]

    redirect_uri = str(request.url_for("google_auth_callback"))
    creds = exchange_code_for_tokens(code, redirect_uri)

    try:
        oauth2 = build("oauth2", "v2", credentials=creds)
        info = oauth2.userinfo().get().execute() or {}
        email = info.get("email")
    except Exception:
        email = None

    TokenStore.save(user_id, {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "scopes": list(creds.scopes or []),
        "expiry": int(getattr(creds, "expiry", 0).timestamp()) if getattr(creds, "expiry", None) else None,
        "email": email,
    })

    return HTMLResponse("""
      <script>
        try { window.opener && window.opener.postMessage({type:"google-auth-complete"},"*"); } catch(e) {}
        window.close();
      </script>
      <p>Done. You can close this window.</p>
    """)


@router.post("/disconnect")
def google_disconnect(request: Request):
    user_id = get_current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not signed in")

    # Best-effort revoke
    try:
        tok = TokenStore.get(user_id) or {}
        t = tok.get("token") or tok.get("refresh_token")
        if t:
            requests.post(
                "https://oauth2.googleapis.com/revoke",
                params={"token": t},
                headers={"content-type": "application/x-www-form-urlencoded"},
                timeout=5,
            )
    except Exception:
        pass

    TokenStore.delete(user_id)
    return JSONResponse({"ok": True})