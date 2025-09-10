# app/api/endpoints/rag.py
from fastapi import APIRouter, Depends, HTTPException
from app.models.schemas import IngestRequest, QueryRequest, AnswerResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])
_rag_service = RAGService()

def get_rag_service() -> RAGService:
    return _rag_service

@router.post("/ingest")
def ingest(payload: IngestRequest, svc: RAGService = Depends(get_rag_service)):
    n = svc.ingest(file_paths=payload.file_paths, urls=payload.urls)
    return {"chunks_indexed": n}

@router.post("/query", response_model=AnswerResponse)
def query(payload: QueryRequest, svc: RAGService = Depends(get_rag_service)):
    q = (payload.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question must not be empty")
    k = payload.top_k or 4
    return svc.answer(q, top_k=k)  # pass top_k through
