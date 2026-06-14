"""
Query Processing Tests
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def get_auth_headers():
    """Get authentication headers for testing"""
    login_data = {
        "email": "user@example.com",
        "password": "password123"
    }

    response = client.post("/api/auth/login", json=login_data)
    token = response.json()["token"]
    return {"Authorization": f"Bearer {token}"}

def test_biomedical_query():
    """Test biomedical query processing"""
    headers = get_auth_headers()

    query_data = {
        "query": "What is the mechanism of CAR-T cell therapy?",
        "max_results": 5
    }

    response = client.post("/api/query", json=query_data, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "confidence_score" in data
    assert "citations" in data
    assert "follow_up_questions" in data
    assert "processing_time" in data
    assert "query_analysis" in data

    # Check data types
    assert isinstance(data["confidence_score"], float)
    assert isinstance(data["citations"], list)
    assert isinstance(data["follow_up_questions"], list)
    assert isinstance(data["processing_time"], float)

def test_query_with_different_topics():
    """Test queries on different biomedical topics"""
    headers = get_auth_headers()

    queries = [
        "How does mRNA vaccine work?",
        "What are the symptoms of Alzheimer's disease?",
        "CRISPR gene editing mechanism",
        "COVID-19 treatment options"
    ]

    for query in queries:
        query_data = {"query": query}
        response = client.post("/api/query", json=query_data, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0
        assert data["confidence_score"] > 0

def test_empty_query():
    """Test empty query handling"""
    headers = get_auth_headers()

    query_data = {"query": ""}
    response = client.post("/api/query", json=query_data, headers=headers)

    # Should handle gracefully
    assert response.status_code in [400, 422]  # Bad request or validation error

def test_query_parameters():
    """Test query with different parameters"""
    headers = get_auth_headers()

    query_data = {
        "query": "diabetes treatment",
        "max_results": 15,
        "include_full_text": True
    }

    response = client.post("/api/query", json=query_data, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data

def test_metrics_endpoint():
    """Test system metrics endpoint"""
    headers = get_auth_headers()

    response = client.get("/api/metrics", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "total_documents" in data
    assert "total_vectors" in data
    assert "elasticsearch_status" in data
    assert "average_response_time" in data
    assert "cache_hit_rate" in data
