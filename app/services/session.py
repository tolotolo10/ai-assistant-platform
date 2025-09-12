import uuid
from fastapi import Request, Response

COOKIE_NAME = "uid"

def get_current_user_id(request: Request) -> str | None:
    return request.cookies.get(COOKIE_NAME)

def ensure_user_id(request: Request, response: Response) -> str:
    uid = request.cookies.get(COOKIE_NAME)
    if not uid:
        uid = str(uuid.uuid4())
        response.set_cookie(
            COOKIE_NAME, uid,
            httponly=True, samesite="lax", secure=True
        )
    return uid
