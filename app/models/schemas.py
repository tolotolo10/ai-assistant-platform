# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class IngestRequest(BaseModel):
    file_paths: Optional[List[str]] = None
    urls: Optional[List[str]] = None

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=4, ge=1, le=20)  # NEW

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
