from __future__ import annotations

from pydantic import BaseModel, Field
from langchain.tools import tool
from typing import List

from app.api.endpoints import rag as rag_router

_rag_service = rag_router.get_rag_service()  


class RagQueryInput(BaseModel):
    question: str = Field(..., description="The user's question to answer using the document index.")
    top_k: int = Field(4, ge=1, le=10, description="How many passages to retrieve (default 4).")


def _format_sources_md(sources: List[dict]) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        meta = s.get("metadata", {}) or {}
        title = meta.get("source") or meta.get("title") or meta.get("file") or "source"
        snippet = (s.get("snippet") or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300] + "…"
        lines.append(f"[{i}] **{title}** – {snippet}")
    return "\n".join(lines) if lines else "_No sources returned_"


@tool("rag_query", args_schema=RagQueryInput)
def rag_query_tool(question: str, top_k: int = 4) -> str:
    """
    Retrieve and answer questions grounded in your indexed documents.
    Returns a concise answer followed by a 'Sources' section with citations.
    """
    result = _rag_service.answer(question, k=top_k)
    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []
    md_sources = _format_sources_md(sources)
    return f"{answer}\n\n---\n**Sources**\n{md_sources}"
