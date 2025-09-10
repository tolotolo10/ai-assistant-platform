from __future__ import annotations
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.services.session import get_current_user_id
from app.tools.real_tools import create_calendar_event_tool

router = APIRouter(prefix="/api/agent", tags=["agent-actions"])

@router.post("/schedule-meeting")
async def schedule_meeting(request: Request):
    user_id = get_current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not signed in")

    body = await request.json()
    body["user_id"] = user_id

    # Call the LangChain Tool directly with a dict payload
    result = create_calendar_event_tool.run(body)

    # Try to parse the tool's structured JSON response
    try:
        js = json.loads(result)
    except Exception:
        return JSONResponse(
            {"message": f"Error from calendar tool: {result}"},
            status_code=500
        )

    if js.get("error") == "AUTH_REQUIRED":
        return JSONResponse({"message": "Please connect your Google account first."}, status_code=401)

    if not js.get("ok"):
        return JSONResponse({"message": "Could not schedule the meeting."}, status_code=500)

    ev = js["event"]
    attendees = ", ".join(a.get("email") for a in ev.get("attendees", [])) or "none"

    # Friendly markdown message for your chat UI
    msg = (
        f"✅ **Meeting scheduled**: {ev.get('summary','(no title)')}\n"
        f"- When: `{ev['start'].get('dateTime')}` → `{ev['end'].get('dateTime')}` "
        f"({ev['start'].get('timeZone')})\n"
        f"- Calendar: {ev.get('htmlLink') or 'n/a'}\n"
        f"- Meet: {ev.get('hangoutLink') or 'n/a'}\n"
        f"- Attendees: {attendees}"
    )

    return JSONResponse({"message": msg, "event": ev})
