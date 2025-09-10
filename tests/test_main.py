"""
Test suite for main application endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestHealthEndpoints:
    """Test health check and basic endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["status"] == "healthy"
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "app_name" in data
        assert "version" in data
    
    def test_api_v1_root(self):
        """Test API v1 root endpoint."""
        response = client.get("/api/v1")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert data["endpoints"]["docs"] == "/docs"

@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test asynchronous endpoint functionality."""
    
    async def test_async_root(self):
        """Test that async endpoints work correctly."""
        response = client.get("/")
        assert response.status_code == 200

class TestErrorHandlers:
    """Test custom error handlers."""
    
    def test_404_handler(self):
        """Test custom 404 error handler."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "Endpoint not found"
        assert data["status_code"] == 404
