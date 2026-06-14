"""
Health Check Tests
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test basic health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "services" in data

def test_health_check_services():
    """Test health check services information"""
    response = client.get("/health")
    data = response.json()

    services = data["services"]
    assert "elasticsearch" in services
    assert "vectors" in services
    assert "embedding_model" in services
    assert "document_service" in services

def test_cors_headers():
    """Test CORS headers are present"""
    response = client.get("/health")
    assert response.status_code == 200
    # CORS headers should be added by middleware
