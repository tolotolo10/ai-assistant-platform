import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.endpoints.rag import get_rag_service
from app.models.schemas import AnswerResponse


class FakeRAGService:
    def __init__(self):
        self.ingested = 0

    def ingest(self, file_paths=None, urls=None):
        self.ingested += 3
        return 3

    def answer(self, question: str):
        return {"answer": f"echo: {question}", "sources": [{"metadata": {"source": "fake"}, "snippet": "lorem"}]}


@pytest.fixture(autouse=True)
def override_rag():
    app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


def test_ingest_minimal():
    resp = client.post("/api/v1/rag/ingest", json={"file_paths": ["README.md"]})
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] == 3


def test_query_basic():
    resp = client.post("/api/v1/rag/query", json={"question": "What is this platform?"})
    assert resp.status_code == 200
    parsed = AnswerResponse(**resp.json())
    assert parsed.answer.startswith("echo:")
    assert len(parsed.sources) == 1


def test_query_validation():
    resp = client.post("/api/v1/rag/query", json={"question": ""})
    assert resp.status_code == 400
