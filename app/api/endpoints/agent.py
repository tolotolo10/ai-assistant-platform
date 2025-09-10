# app/api/endpoints/agent.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.agent_service import AgentService

router = APIRouter(prefix="/agent", tags=["Agent"])
agent_service = AgentService()


class AgentQuery(BaseModel):
    message: str
    session_id: str


@router.post("/query")
def query(payload: AgentQuery):
    msg = (payload.message or "").strip()
    sid = (payload.session_id or "").strip()
    if not msg:
        raise HTTPException(400, "message must not be empty")
    if not sid:
        raise HTTPException(400, "session_id must not be empty")
    try:
        answer = agent_service.ask(sid, msg)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(500, f"agent error: {e}")


@router.get("/stream")
def stream(
    message: str = Query(..., min_length=1),
    session_id: str = Query(..., min_length=1),
):
    async def gen():
        async for frame in agent_service.astream(session_id, message):
            yield frame

    return StreamingResponse(gen(), media_type="text/event-stream")
