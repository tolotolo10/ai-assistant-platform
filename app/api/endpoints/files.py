# app/api/endpoints/files.py
from __future__ import annotations

import os
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.api.endpoints import rag as rag_router  # reuse the same RAG service singleton

router = APIRouter(prefix="/files", tags=["Files"])

# get the shared RAGService instance from rag.py
_rag_service = rag_router.get_rag_service()

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Accept one or more files (PDF/TXT), save them to ./data/uploads,
    and ingest them into the vector store.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    base_dir = os.path.join(os.getcwd(), "data", "uploads")
    os.makedirs(base_dir, exist_ok=True)

    saved_paths: List[str] = []
    for f in files:
        # Basic content-type guard (octet-stream is common from browsers)
        if f.content_type not in {
            "application/pdf",
            "text/plain",
            "application/octet-stream",
        }:
            raise HTTPException(status_code=400, detail=f"Unsupported type: {f.content_type}")

        out_path = os.path.join(base_dir, f.filename)
        with open(out_path, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                out.write(chunk)
        saved_paths.append(out_path)

    # Ingest into the RAG index
    n_chunks = _rag_service.ingest(file_paths=saved_paths, urls=None)
    return {"saved": saved_paths, "chunks_indexed": n_chunks}
