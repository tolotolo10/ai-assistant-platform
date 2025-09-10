# app/tools/basic_tools.py
from langchain.tools import tool
import re
from typing import List

# --- Helpers ---------------------------------------------------------------

def _kv_parse(spec: str) -> dict:
    """
    Parse a simple semi-colon separated spec into a dict.
    Example:
      "to=alice@example.com; subject=Hi; body=Hello there"
    """
    out = {}
    for part in spec.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out

def _list_parse(csv: str) -> List[str]:
    return [p.strip() for p in csv.split(",") if p.strip()]

# --- Tools (single-string input only) -------------------------------------

@tool("get_weather")
def get_weather(location: str) -> str:
    """Get current weather for a location. Input: 'City, Country'."""
    fake_weather = {
        "Paris, France": "Cloudy, 18°C",
        "New York, USA": "Sunny, 25°C",
    }
    loc = location.strip()
    return fake_weather.get(loc, f"Weather data not available for {loc}")

@tool("send_email")
def send_email(spec: str) -> str:
    """
    Send an email (dummy). Input format (single string):
      'to=alice@example.com; subject=Status; body=We are on track.'
    """
    data = _kv_parse(spec)
    to = data.get("to")
    subject = data.get("subject", "(no subject)")
    body = data.get("body", "(empty)")
    if not to:
        return "Error: missing 'to=' in send_email input."
    return f"📧 Email prepared for {to} | Subject: {subject} | Body: {body}"

@tool("schedule_meeting")
def schedule_meeting(spec: str) -> str:
    """
    Schedule a meeting (dummy). Input format (single string):
      'date=2025-09-07; time=10:00; participants=bob@example.com,jane@example.com; description=Q3 plan'
    """
    data = _kv_parse(spec)
    date = data.get("date")
    time = data.get("time")
    participants = _list_parse(data.get("participants", ""))
    description = data.get("description", "")
    if not date or not time or not participants:
        return "Error: need 'date', 'time' and 'participants' in schedule_meeting input."
    return (
        f"📅 Meeting scheduled on {date} at {time} with {', '.join(participants)}. "
        f"Description: {description or '(none)'}"
    )
