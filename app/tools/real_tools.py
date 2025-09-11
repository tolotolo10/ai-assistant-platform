# app/tools/real_tools.py
from __future__ import annotations

import os
import ssl
import smtplib
import datetime as dt
from email.mime.text import MIMEText
from typing import Optional, List

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote

from pydantic import BaseModel, Field, validator
from langchain.tools import tool
import json


import uuid

from app.services.rag_service import RAGService

# Singleton RAG service for the process
_RAG_SVC: RAGService | None = None


# =========================
# Helpers / ENV
# =========================
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")


# =========================
# Location + Time helpers
# =========================
@tool("get_location_from_ip")
def get_location_from_ip() -> str:
    """Get a rough geographical location based on the public IP address."""
    try:
        r = requests.get("https://ipinfo.io/json", timeout=10)
        r.raise_for_status()
        data = r.json()
        if "loc" in data:
            lat, lon = (data.get("loc") or ",").split(",")
            city = data.get("city") or "N/A"
            country = data.get("country") or "N/A"
            return (
                f"Latitude: {lat}\nLongitude: {lon}\n"
                f"City: {city}\nCountry: {country}"
            )
        return "Location could not be determined."
    except Exception as e:
        return f"Error occurred: {e}"


@tool("get_current_datetime")
def get_current_datetime() -> str:
    """Return the current local datetime in ISO 8601."""
    return dt.datetime.now().isoformat(timespec="seconds")


# =========================
# SerpAPI – Google Weather
# =========================
class WeatherInput(BaseModel):
    location: Optional[str] = Field(
        None,
        description="City and country, e.g. 'Dubai, UAE'. If not provided, tool will try to infer from IP.",
    )

@tool("get_weather_serp", args_schema=WeatherInput)
def get_weather_serp_tool(location: Optional[str] = None) -> str:
    """
    Get current weather using SerpAPI's Google Search.
    - If location not provided, approximates from IP.
    - Parses Google's weather answer box.
    """
    if not SERPAPI_API_KEY:
        return "SERPAPI_API_KEY is not set. Please add it to your environment."

    try:
        # Infer location if missing
        if not location:
            ip = requests.get("https://ipinfo.io/json", timeout=10)
            ip.raise_for_status()
            city = ip.json().get("city")
            country = ip.json().get("country")
            if city and country:
                location = f"{city}, {country}"
            elif city:
                location = city
            else:
                location = "your location"

        params = {
            "engine": "google",
            "q": f"weather in {location}",
            "api_key": SERPAPI_API_KEY,
        }
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=12)
        resp.raise_for_status()
        js = resp.json()

        # Prefer answer_box weather
        ab = js.get("answer_box", {})
        # Common fields in SerpAPI's weather answer box
        temp = ab.get("temperature") or ab.get("temperature_f") or ab.get("temperature_c")
        unit = "°C" if ab.get("temperature") or ab.get("temperature_c") else "°F"
        condition = ab.get("weather") or ab.get("description") or ab.get("snippet")
        loc_name = ab.get("location") or location
        humidity = ab.get("humidity")
        wind = ab.get("wind")

        if temp is not None or condition:
            parts = [f"Weather in {loc_name}"]
            if temp is not None:
                parts.append(f"{temp}{unit}")
            if condition:
                parts.append(condition)
            if humidity:
                parts.append(f"Humidity {humidity}")
            if wind:
                parts.append(f"Wind {wind}")
            return ", ".join(parts) + "."

        # Fallback: organic result snippet
        organic = js.get("organic_results") or []
        if organic:
            title = organic[0].get("title", "Weather result")
            snippet = organic[0].get("snippet", "")
            return f"{title}: {snippet}"

        return f"Couldn't parse weather for {location}."
    except Exception as e:
        return f"Weather lookup failed: {e}"


# =========================
# SerpAPI – Google Search
# =========================
class SearchInput(BaseModel):
    query: str = Field(..., description="Google search query")

@tool("google_search", args_schema=SearchInput)
def google_search_tool(query: str) -> str:
    """
    Search Google via SerpAPI and return a concise, 1–3 sentence answer
    synthesized from top results (not just links).
    """
    if not SERPAPI_API_KEY:
        return "SERPAPI_API_KEY is not set. Please add it to your environment."

    try:
        params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=12)
        r.raise_for_status()
        js = r.json()

        # If answer_box has a direct answer, use it
        ab = js.get("answer_box") or {}
        direct = ab.get("answer") or ab.get("snippet") or ab.get("highlighted_snippets")
        if isinstance(direct, list) and direct:
            direct = direct[0]
        if direct:
            return str(direct)

        # Otherwise compose from the first couple of organic results
        org = js.get("organic_results") or []
        if not org:
            return "No results found."

        snippets = []
        for res in org[:3]:
            title = res.get("title", "").strip()
            snippet = res.get("snippet", "").strip()
            if snippet:
                snippets.append(snippet)

        if snippets:
            # Summarize very briefly
            text = " ".join(snippets)
            sentences = text.split(". ")
            summary = ". ".join(sentences[:2]).strip()
            if not summary.endswith("."):
                summary += "."
            return summary

        return org[0].get("title", "Found a result but no snippet.")
    except Exception as e:
        return f"Search failed: {e}"


# =========================
# Email via Gmail SMTP (unchanged)
# =========================
class EmailInput(BaseModel):
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Plain text body")

@tool("send_email", args_schema=EmailInput)
def send_email_smtp_tool(to: str, subject: str, body: str) -> str:
    """
    Send an email via Gmail SMTP. Requires env:
      GMAIL_ADDRESS, GMAIL_APP_PASSWORD (2FA App Password).
    """
    gmail = os.getenv("GMAIL_ADDRESS")
    app_pw = os.getenv("GMAIL_APP_PASSWORD")
    if not gmail or not app_pw:
        return ("Email not configured. Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD env vars. "
                "Use a Gmail App Password (2FA required).")

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = gmail
    msg["To"] = to
    msg["Subject"] = subject

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(gmail, app_pw)
            server.sendmail(gmail, [to], msg.as_string())
        return f"Email sent to {to}."
    except Exception as e:
        return f"Failed to send email: {e}"


# =========================
# Google Calendar (unchanged)
# =========================
def _get_google_creds():
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        import pathlib

        SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
        root = pathlib.Path(__file__).resolve().parents[2]
        gdir = root / "google"
        gdir.mkdir(exist_ok=True)
        token_path = gdir / "token.json"
        cred_path = gdir / "credentials.json"

        creds = None
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not cred_path.exists():
                    return None, "Missing google/credentials.json (OAuth client)."
                flow = InstalledAppFlow.from_client_secrets_file(str(cred_path), SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, "w") as f:
                f.write(creds.to_json())
        return creds, None
    except Exception as e:
        return None, f"Google auth error: {e}"


class MeetingInput(BaseModel):
    user_id: str = Field(..., description="Current app user id")
    title: str = Field(..., description="Meeting title")
    start_iso: str = Field(..., description="Start datetime ISO 8601 (e.g. 2025-09-07T10:00:00)")
    end_iso: str | None = Field(None, description="End datetime ISO 8601 (defaults to +1 hour)")
    timezone: str = Field("UTC", description="IANA timezone, e.g. 'Europe/Paris'")
    attendees: list[str] | None = Field(None, description="List of attendee emails")
    description: str = Field("", description="Optional description")

    @validator("end_iso", always=True)
    def default_end(cls, v, values):
        if v:
            return v
        try:
            start = dt.datetime.fromisoformat(values["start_iso"])
            return (start + dt.timedelta(hours=1)).isoformat()
        except Exception:
            return values.get("start_iso")

@tool("create_calendar_event", args_schema=MeetingInput)
def create_calendar_event_tool(
    user_id: str,
    title: str,
    start_iso: str,
    end_iso: str | None = None,
    timezone: str = "UTC",
    attendees: list[str] | None = None,
    description: str = "",
) -> str:
    """
    Create a meeting in the authenticated user's Google Calendar, send invites,
    and return a JSON payload. If the user hasn't connected Google yet, returns
    {"error": "AUTH_REQUIRED"}.
    """
    try:
        from googleapiclient.discovery import build
        from app.tools.google_creds import get_user_google_creds

        creds, err = get_user_google_creds(user_id)
        if err == "AUTH_REQUIRED":
            return json.dumps({"error": "AUTH_REQUIRED"}, ensure_ascii=False)
        if err:
            return json.dumps({"error": f"Calendar not configured: {err}"}, ensure_ascii=False)

        service = build("calendar", "v3", credentials=creds)

        if not end_iso:
            start_dt = dt.datetime.fromisoformat(start_iso)
            end_iso = (start_dt + dt.timedelta(hours=1)).isoformat()

        event = {
            "summary": title,
            "description": description,
            "start": {"dateTime": start_iso, "timeZone": timezone},
            "end": {"dateTime": end_iso, "timeZone": timezone},
            "conferenceData": {
                "createRequest": {
                    "requestId": str(uuid.uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
        }
        if attendees:
            event["attendees"] = [{"email": a} for a in attendees]

        created = service.events().insert(
            calendarId="primary",
            body=event,
            conferenceDataVersion=1,
            sendUpdates="all",   # send email invitations
        ).execute()

        payload = {
            "id": created.get("id"),
            "htmlLink": created.get("htmlLink"),
            "hangoutLink": created.get("hangoutLink")
                or created.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri"),
            "summary": created.get("summary"),
            "start": created.get("start"),
            "end": created.get("end"),
            "attendees": created.get("attendees", []),
        }
        return json.dumps({"ok": True, "event": payload}, ensure_ascii=False)

    except ModuleNotFoundError as e:
        # helpful message if google client libs aren't installed
        return json.dumps({
            "error": f"Missing package: {e}. Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to create calendar event: {e}"}, ensure_ascii=False)

# pydantic v1/v2 compatibility
try:
    from pydantic import BaseModel, Field, ConfigDict
    _MODEL_CONFIG = dict(model_config=ConfigDict(extra="allow"))  # v2
except Exception:
    from pydantic import BaseModel, Field
    class _Cfg:  # v1
        extra = "allow"
    _MODEL_CONFIG = dict(Config=_Cfg)

from langchain.tools import tool

from app.services.rag_service import RAGService

_RAG_SVC: RAGService | None = None  # module-level singleton

class RagQueryInput(BaseModel):
    question: str = Field(..., description="User question for the private KB")
    top_k: int = Field(4, description="How many chunks to retrieve")
    # apply compat config
    if "model_config" in _MODEL_CONFIG:
        model_config = _MODEL_CONFIG["model_config"]
    else:
        class Config(_MODEL_CONFIG["Config"]):  # type: ignore[name-defined]
            pass

@tool("rag_query", args_schema=RagQueryInput)
def rag_query_tool(question: str, top_k: int = 4) -> str:
    """
    Answer using the private knowledge base (RAG).
    """
    global _RAG_SVC
    if _RAG_SVC is None:
        _RAG_SVC = RAGService()

    try:
        result = _RAG_SVC.answer(question, top_k=top_k)
        answer = (result.get("answer") or "").strip()
        sources = result.get("sources") or []

        seen = set()
        source_lines = []
        for s in sources[:8]:
            meta = s.get("metadata") or {}
            title = (
                meta.get("source")
                or meta.get("title")
                or meta.get("file_path")
                or meta.get("url")
                or "document"
            )
            if title not in seen:
                source_lines.append(f"- {title}")
                seen.add(title)

        if not answer:
            return "No answer found in the knowledge base."

        if source_lines:
            return answer + "\n\nSources:\n" + "\n".join(source_lines)
        return answer
    except Exception as e:
        return f"RAG error: {e}"

# --- Final Answer (structured) -----------------------------------------------


class FinalAnswerInput(BaseModel):
    answer: str = Field(..., description="Natural-language answer for the user.")
    tools_used: List[str] = Field(
        default_factory=list,
        description="List of tool names actually used in this run."
    )

@tool("final_answer", args_schema=FinalAnswerInput)
def final_answer_tool(answer: str, tools_used: list[str]) -> str:
    """
    Use this tool to provide a final answer to the user.
    The answer should be in natural language as this will be provided
    to the user directly. The tools_used must include a list of tool
    names that were used within the `scratchpad`.
    """
    # Return JSON so downstream code can parse it easily.
    return json.dumps({"answer": answer, "tools_used": tools_used}, ensure_ascii=False)

# =========================
# List calendar events tool
# =========================
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from zoneinfo import ZoneInfo

class EventQueryInput(BaseModel):
    user_id: Optional[str] = Field(
        None, description="Current app user id. DEV: if omitted, uses the only connected account."
    )
    timeframe: Optional[Literal["this_week", "this_month", "next_7_days", "next_30_days"]] = None
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    timezone: str = Field("UTC", description="IANA timezone, e.g. 'Europe/Paris'")
    include_canceled: bool = False
    max_results: int = Field(100, ge=1, le=2500)

    @validator("timezone")
    def validate_tz(cls, v):
        try:
            ZoneInfo(v)
            return v
        except Exception:
            raise ValueError(f"Invalid timezone: {v}")

    @validator("end_iso")
    def end_after_start(cls, v, values):
        s = values.get("start_iso")
        if v and s:
            try:
                sdt = dt.datetime.fromisoformat(s)
                edt = dt.datetime.fromisoformat(v)
                if edt <= sdt:
                    raise ValueError("end_iso must be after start_iso")
            except Exception:
                # If parsing fails, let Google API error later; we keep this gentle.
                return v
        return v


def _month_bounds(now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end

def _week_bounds(now: dt.datetime) -> tuple[dt.datetime, dt.datetime]:
    # ISO week: Monday 00:00 through next Monday 00:00 (Europe typically expects Monday start)
    monday = (now - dt.timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    return monday, monday + dt.timedelta(days=7)

def _window_from_timeframe(tf: str, tz: ZoneInfo) -> tuple[str, str]:
    now = dt.datetime.now(tz)
    if tf == "this_month":
        s, e = _month_bounds(now)
    elif tf == "next_7_days":
        s = now.replace(microsecond=0)
        e = s + dt.timedelta(days=7)
    elif tf == "next_30_days":
        s = now.replace(microsecond=0)
        e = s + dt.timedelta(days=30)
    else:  # default "this_week"
        s, e = _week_bounds(now)
    return s.isoformat(), e.isoformat()


@tool("list_calendar_events", args_schema=EventQueryInput)
def list_calendar_events_tool(
    user_id: str | None = None,   # ← make optional
    timeframe: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
    timezone: str = "UTC",
    include_canceled: bool = False,
    max_results: int = 100,
) -> str:
    """
    List Google Calendar events for a user between a time window.
    Returns JSON:
      - {"error": "AUTH_REQUIRED"} if the user hasn't connected Google yet.
      - {"ok": True, "timeframe": <...>, "timezone": <...>, "start": <iso>, "end": <iso>,
         "count": <int>, "events": [<google event subset>]}
    Notes:
      - Uses 'primary' calendar.
      - timeMax is exclusive, as per Google Calendar API semantics.
      - Includes all-day events (which use 'date' rather than 'dateTime').
    """
    try:
        from googleapiclient.discovery import build
        from app.tools.google_creds import get_user_google_creds

        tz = ZoneInfo(timezone)

        # Resolve auth
        creds, err = get_user_google_creds(user_id)
        if err == "AUTH_REQUIRED":
            return json.dumps({"error": "AUTH_REQUIRED"}, ensure_ascii=False)
        if err:
            return json.dumps({"error": f"Calendar not configured: {err}"}, ensure_ascii=False)

        # Resolve window
        if not (start_iso and end_iso):
            tf = timeframe or "this_week"
            start_iso, end_iso = _window_from_timeframe(tf, tz)
        else:
            tf = "custom"

        service = build("calendar", "v3", credentials=creds)

        # Fetch list
        resp = service.events().list(
            calendarId="primary",
            timeMin=start_iso,
            timeMax=end_iso,
            singleEvents=True,      # expand recurring to instances
            orderBy="startTime",
            maxResults=max_results,
            showDeleted=include_canceled,
        ).execute()

        items = resp.get("items", []) or []

        # Filter cancelled unless caller asked otherwise
        if not include_canceled:
            items = [ev for ev in items if ev.get("status") != "cancelled"]

        # Trim to a tidy subset per event
        out = []
        for ev in items:
            out.append({
                "id": ev.get("id"),
                "htmlLink": ev.get("htmlLink"),
                "status": ev.get("status"),
                "summary": ev.get("summary"),
                "description": ev.get("description"),
                "start": ev.get("start"),  # may contain 'date' or 'dateTime'
                "end": ev.get("end"),
                "location": ev.get("location"),
                "attendees": ev.get("attendees", []),
                "organizer": ev.get("organizer", {}),
                "hangoutLink": ev.get("hangoutLink") or ev.get("conferenceData", {}).get("entryPoints", [{}])[0].get("uri"),
            })

        payload = {
            "ok": True,
            "timeframe": tf,
            "timezone": timezone,
            "start": start_iso,
            "end": end_iso,
            "count": len(out),
            "events": out,
        }
        return json.dumps(payload, ensure_ascii=False)

    except ModuleNotFoundError as e:
        return json.dumps({
            "error": f"Missing package: {e}. Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to list calendar events: {e}"}, ensure_ascii=False)

# =========================
# Tool registry
# =========================
def get_all_tools():
    """
    Return the list of Tool objects the agent can call.
    Prefer SerpAPI-based weather and Google search.
    """
    return [
        get_location_from_ip,
        get_current_datetime,
        get_weather_serp_tool,   # weather via SerpAPI
        google_search_tool,      # Google search via SerpAPI
        send_email_smtp_tool,
        list_calendar_events_tool,
        create_calendar_event_tool,
        rag_query_tool,
        final_answer_tool,
    ]
