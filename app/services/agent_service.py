# app/services/agent_service.py
from __future__ import annotations
from typing import Dict, Any, AsyncIterator, Tuple

import time
import string

from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.services.memory_store import MemoryStore
from app.tools.real_tools import get_all_tools
from app.config import settings


def sse_event(event: str, data: str) -> str:
    """Format a server-sent event (SSE) frame."""
    return f"event: {event}\ndata: {data}\n\n"


ALNUM = set(string.ascii_letters + string.digits)


def _smart_join(prev: str, piece: str) -> str:
    """Join streamed chunks while fixing line-break artifacts."""
    if not piece:
        return prev
    piece = piece.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    if prev and piece and (prev[-1] in ALNUM) and (piece[0] in ALNUM):
        piece = " " + piece
    return prev + piece


class AgentService:
    """
    Session-aware agent service (in-memory only).
    - One memory + agent executor per session id
    - Token streaming via astream_events
    """

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=settings.temperature,
            openai_api_key=settings.openai_api_key,
            streaming=True,
        )

        self.tools = get_all_tools()
        self.mem_store = MemoryStore(k=10)
        self._sessions: Dict[str, Tuple[AgentExecutor, Any]] = {}

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a helpful assistant. Use tools when they improve accuracy or speed. "
                    "After a tool runs, its result will appear in the scratchpad. "
                    "Use the 'rag_query' tool for questions about uploaded documents. "
                    "When you are ready to answer, ALWAYS call the 'final_answer' tool exactly once "
                    "with the final natural-language answer and a list of the tools you actually used."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def _get_executor(self, session_id: str) -> AgentExecutor:
        if session_id in self._sessions:
            return self._sessions[session_id][0]

        memory = self.mem_store.get(session_id)
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            early_stopping_method="force",
            return_intermediate_steps=False,
        )
        self._sessions[session_id] = (executor, memory)
        return executor

    # ----- sync -----
    def ask(self, session_id: str, message: str) -> str:
        ex = self._get_executor(session_id)
        result: Dict[str, Any] = ex.invoke({"input": message})
        return result.get("output", "")

    # ----- streaming -----
    async def astream(self, session_id: str, message: str) -> AsyncIterator[str]:
        ex = self._get_executor(session_id)

        buf = ""
        last_flush = time.monotonic()

        try:
            async for ev in ex.astream_events({"input": message}, version="v1"):
                kind = ev.get("event")

                if kind == "on_chat_model_stream":
                    chunk = ev.get("data", {}).get("chunk")
                    if chunk and getattr(chunk, "content", None):
                        buf = _smart_join(buf, chunk.content)
                        now = time.monotonic()
                        if len(buf) >= 64 or (now - last_flush) >= 0.07:
                            yield sse_event("delta", buf)
                            buf = ""
                            last_flush = now

                elif kind == "on_tool_start":
                    name = ev.get("name") or ev.get("data", {}).get("name")
                    if name:
                        yield sse_event("delta", f"\n[tool:{name}…] ")
                elif kind == "on_tool_end":
                    name = ev.get("name") or ev.get("data", {}).get("name")
                    if name:
                        yield sse_event("delta", f" [/tool:{name}]\n")

        except Exception as e:
            if buf:
                yield sse_event("delta", buf)
            yield sse_event("error", str(e))
            return

        if buf:
            yield sse_event("delta", buf)
        yield sse_event("done", "ok")

    def drop_session(self, session_id: str) -> None:
        self.mem_store.drop(session_id)
        self._sessions.pop(session_id, None)
