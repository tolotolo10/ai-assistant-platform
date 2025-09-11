from __future__ import annotations
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from app.services.session import get_current_user_id
from app.tools.real_tools import create_calendar_event_tool, list_calendar_events_tool

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



@router.get("/calendar/peek")
async def calendar_peek(request: Request, timeframe: str = "this_week", timezone: str = "Europe/Paris"):
    """
    Quick endpoint your UI can call to show a small badge like '3 events this week'.
    Returns 401 if Google not connected (so you can prompt the user to connect).
    """
    user_id = get_current_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not signed in")

    payload = {
        "user_id": user_id,
        "timeframe": timeframe,
        "timezone": timezone,
        "max_results": 50,
    }
    result = list_calendar_events_tool.run(payload)
    try:
        js = json.loads(result)
    except Exception:
        return JSONResponse({"message": f"Error from calendar tool: {result}"}, status_code=500)

    if js.get("error") == "AUTH_REQUIRED":
        return JSONResponse({"message": "Please connect your Google account first."}, status_code=401)

    if not js.get("ok"):
        return JSONResponse({"message": "Could not list your events."}, status_code=500)

    # Good minimal response for UI badges
    return JSONResponse({
        "timeframe": js["timeframe"],
        "start": js["start"],
        "end": js["end"],
        "count": js["count"],
        "events": js["events"][:5],   # cap preview
    })
