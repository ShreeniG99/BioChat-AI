"""
Authentication Tests
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_demo_login():
    """Test demo user login"""
    login_data = {
        "email": "user@example.com",
        "password": "password123"
    }

    response = client.post("/api/auth/login", json=login_data)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] == True
    assert "token" in data
    assert "user" in data
    assert data["user"]["email"] == "user@example.com"

def test_invalid_login():
    """Test invalid login credentials"""
    login_data = {
        "email": "invalid@example.com", 
        "password": "wrongpassword"
    }

    response = client.post("/api/auth/login", json=login_data)
    assert response.status_code == 401

def test_signup():
    """Test user signup"""
    signup_data = {
        "email": "test@example.com",
        "password": "testpass123",
        "confirm_password": "testpass123"
    }

    response = client.post("/api/auth/signup", json=signup_data)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] == True
    assert "token" in data
    assert "user" in data

def test_signup_password_mismatch():
    """Test signup with password mismatch"""
    signup_data = {
        "email": "test2@example.com",
        "password": "testpass123",
        "confirm_password": "different123"
    }

    response = client.post("/api/auth/signup", json=signup_data)
    assert response.status_code == 400

def test_protected_endpoint_no_token():
    """Test accessing protected endpoint without token"""
    response = client.get("/api/history")
    assert response.status_code == 403  # No authorization header

def test_protected_endpoint_with_token():
    """Test accessing protected endpoint with valid token"""
    # First login to get token
    login_data = {
        "email": "user@example.com",
        "password": "password123"
    }

    login_response = client.post("/api/auth/login", json=login_data)
    token = login_response.json()["token"]

    # Use token to access protected endpoint
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/history", headers=headers)

    assert response.status_code == 200
