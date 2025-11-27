# app/services/memory_store.py
from __future__ import annotations
from typing import Dict
from langchain.memory import ConversationBufferWindowMemory


class MemoryStore:
    def __init__(self, k: int = 10):
        self.k = k
        self._store: Dict[str, ConversationBufferWindowMemory] = {}

    def get(self, session_id: str) -> ConversationBufferWindowMemory:
        if session_id not in self._store:
            self._store[session_id] = ConversationBufferWindowMemory(
                k=self.k,
                memory_key="chat_history",
                return_messages=True,
                output_key="output",  # ← add this to remove the warning
            )
        return self._store[session_id]

    def drop(self, session_id: str) -> None:
        """Remove a session's memory from the store"""
        self._store.pop(session_id, None)

    def clear(self, session_id: str) -> None:
        """Clear a session's memory but keep it in the store"""
        if session_id in self._store:
            self._store[session_id].clear()

