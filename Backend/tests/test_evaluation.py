"""
Evaluation Tests
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

def test_evaluation_endpoint():
    """Test evaluation endpoint"""
    headers = get_auth_headers()

    eval_data = {
        "query": "What is hypertension?",
        "question_type": "factoid"
    }

    response = client.post("/api/evaluate/compare", json=eval_data, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "query" in data
    assert "evaluation_metrics" in data
    assert "overall_score" in data

def test_evaluation_with_expected_answer():
    """Test evaluation with expected answer"""
    headers = get_auth_headers()

    eval_data = {
        "query": "What causes diabetes?",
        "expected_answer": "Diabetes is caused by insufficient insulin production or insulin resistance",
        "question_type": "factoid"
    }

    response = client.post("/api/evaluate/compare", json=eval_data, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "evaluation_metrics" in data

def test_different_question_types():
    """Test evaluation for different question types"""
    headers = get_auth_headers()

    test_cases = [
        {
            "query": "Is aspirin effective for heart disease prevention?",
            "question_type": "yesno"
        },
        {
            "query": "List the symptoms of COVID-19",
            "question_type": "list" 
        },
        {
            "query": "Summarize the current understanding of Alzheimer's disease",
            "question_type": "summary"
        }
    ]

    for test_case in test_cases:
        response = client.post("/api/evaluate/compare", json=test_case, headers=headers)
        assert response.status_code == 200

        data = response.json()
        assert data["query_type"] == test_case["question_type"]

def test_evaluation_system_comparison():
    """Test system comparison in evaluation"""
    headers = get_auth_headers()

    eval_data = {
        "query": "How does insulin work?",
        "question_type": "factoid"
    }

    response = client.post("/api/evaluate/compare", json=eval_data, headers=headers)
    data = response.json()

    assert "system_comparison" in data["evaluation_metrics"]
    comparison = data["evaluation_metrics"]["system_comparison"]

    assert "our_system_scores" in comparison
    assert "baseline_comparisons" in comparison
    assert "overall_ranking" in comparison
