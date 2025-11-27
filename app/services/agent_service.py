from typing import Dict, Any, AsyncIterator, Tuple
import time, string
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.services.memory_store import MemoryStore
from app.tools.real_tools import get_all_tools
from app.config import settings

def sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"

ALNUM = set(string.ascii_letters + string.digits)

def _smart_join(prev: str, piece: str) -> str:
    if not piece: return prev
    piece = piece.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    if prev and piece and (prev[-1] in ALNUM) and (piece[0] in ALNUM):
        piece = " " + piece
    return prev + piece


class AgentService:
    """
    Session-aware agent service.
    - Rebuilds AgentExecutor automatically if the toolset changes (prevents stale tools).
    """
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.default_model,
            temperature=settings.temperature,
            openai_api_key=settings.openai_api_key,
            streaming=True,
        )
        self.mem_store = MemoryStore(k=10)
        # session_id -> (executor, memory, tool_signature)
        self._sessions: Dict[str, Tuple[AgentExecutor, Any, str]] = {}

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a helpful assistant. Use tools when they improve accuracy or speed. "
                    "After a tool runs, its result will appear in the scratchpad. "
                    "IMPORTANT: When the user asks questions that could relate to documents (like 'summarize', 'analyze', 'what does it say', etc.), "
                    "ALWAYS use the 'rag_query' tool first to search for relevant information from uploaded documents. "
                    "If the user has recently uploaded documents in this conversation, assume their questions are about those documents. "
                    "Use the 'rag_query' tool for ANY questions about content, summaries, analysis, or information that could be in uploaded documents. "
                    "For calendar questions like 'Do I have a meeting today/this week/this month?', "
                    "use 'list_calendar_events' with an appropriate timeframe. "
                    "For scheduling, use 'create_calendar_event'. "
                    # ⬇️ Escape the braces so LangChain doesn't think {error} is a variable
                    "If any calendar tool returns {{\"error\":\"AUTH_REQUIRED\"}}, DO NOT invent details—"
                    "politely ask the user to connect Google Calendar first. "
                    "When you are ready to answer, ALWAYS call the 'final_answer' tool exactly once "
                    "with the final natural-language answer and a list of the tools you actually used."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

    def _current_tools_and_sig(self):
        tools = get_all_tools()  # always fetch fresh
        names = [getattr(t, "name", repr(t)) for t in tools]
        sig = "|".join(sorted(names))  # stable signature
        return tools, sig

    def _build_executor(self, tools, session_id: str) -> Tuple[AgentExecutor, Any]:
        memory = self.mem_store.get(session_id)
        agent = create_tool_calling_agent(self.llm, tools, self.prompt)
        ex = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            early_stopping_method="force",
            return_intermediate_steps=False,
        )
        return ex, memory

    def _get_executor(self, session_id: str) -> AgentExecutor:
        tools, sig = self._current_tools_and_sig()
        if session_id in self._sessions:
            ex, mem, prev_sig = self._sessions[session_id]
            if prev_sig == sig:
                return ex
        ex, mem = self._build_executor(tools, session_id)
        self._sessions[session_id] = (ex, mem, sig)
        return ex

    def ask(self, session_id: str, message: str) -> str:
        ex = self._get_executor(session_id)
        result: Dict[str, Any] = ex.invoke({"input": message})
        return result.get("output", "")

    async def astream(self, session_id: str, message: str) -> AsyncIterator[str]:
        ex = self._get_executor(session_id)

        buf, last_flush = "", time.monotonic()
        try:
            async for ev in ex.astream_events({"input": message}, version="v1"):
                kind = ev.get("event")
                if kind == "on_chat_model_stream":
                    chunk = ev.get("data", {}).get("chunk")
                    if chunk and getattr(chunk, "content", None):
                        buf = _smart_join(buf, chunk.content)
                        now = time.monotonic()
                        if len(buf) >= 50 or (now - last_flush) >= 0.05:
                            yield sse_event("delta", buf); buf = ""; last_flush = now
                # Tool events are now silently ignored - no tool usage shown to user
        except Exception as e:
            if buf: yield sse_event("delta", buf)
            yield sse_event("error", str(e)); return

        if buf: yield sse_event("delta", buf)
        yield sse_event("done", "ok")

    def drop_session(self, session_id: str) -> None:
        self.mem_store.drop(session_id)
        self._sessions.pop(session_id, None)

    def clear_session(self, session_id: str) -> None:
        """Clear the memory for a specific session"""
        self.mem_store.clear(session_id)

    def drop_all_sessions(self) -> None:
        for sid in list(self._sessions.keys()):
            self.drop_session(sid)
