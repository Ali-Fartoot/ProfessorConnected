import pytest
import requests
import os
from unittest.mock import patch, MagicMock

BASE_URL = "http://localhost:8000"

# Fixture for cleanup before and after tests
@pytest.fixture(autouse=True)
def setup_and_cleanup():
    # Setup - clean the database before tests
    requests.delete(f"{BASE_URL}/cleanup_database")
    yield
    # Cleanup after tests
    requests.delete(f"{BASE_URL}/cleanup_database")

@pytest.mark.integration
def test_add_professor_success():
    """Test successful professor addition"""
    professor_name = "Nathan Lambert"
    response = requests.post(
        f"{BASE_URL}/add_professor",
        json={
            "professor_name": professor_name,
            "number_of_articles": 3
        }
    )
    
    assert response.status_code == 200
    assert f"Successfully added professor {professor_name}" in response.json()["message"]

@pytest.mark.integration
def test_add_professor_failure():
    """Test professor addition failure"""
    professor_name = "NonexistentProfessor"
    response = requests.post(
        f"{BASE_URL}/add_professor",
        json={
            "professor_name": professor_name,
            "number_of_articles": -1  # Invalid number to trigger failure
        }
    )
    
    assert response.status_code in [400, 500]  # Accept either client or server error

@pytest.mark.integration
def test_search_functionality():
    """Test search endpoint"""
    # First add a professor
    professor_name = "Nathan Lambert"
    add_response = requests.post(
        f"{BASE_URL}/add_professor",
        json={
            "professor_name": professor_name,
            "number_of_articles": 3
        }
    )
    assert add_response.status_code == 200

    # Then search
    search_response = requests.post(
        f"{BASE_URL}/search",
        json={
            "professor_name": professor_name,
            "limit": 5
        }
    )
    
    assert search_response.status_code == 200
    assert "results" in search_response.json()

@pytest.mark.integration
def test_visualization():
    """Test visualization endpoint"""
    # First add a professor
    professor_name = "Nathan Lambert"
    requests.post(
        f"{BASE_URL}/add_professor",
        json={
            "professor_name": professor_name,
            "number_of_articles": 3
        }
    )

    response = requests.post(
        f"{BASE_URL}/search_with_visualization",
        json={
            "professor_name": professor_name
        }
    )
    
    assert response.status_code == 200
    assert "network_image" in response.json()

@pytest.mark.integration
def test_database_cleanup():
    """Test database cleanup functionality"""
    response = requests.delete(f"{BASE_URL}/cleanup_database")
    assert response.status_code == 200
    assert "successfully" in response.json()["message"].lower()

@pytest.mark.integration
def test_full_workflow():
    """Test the entire workflow"""
    professor_name = "Nathan Lambert"

    # 1. Add professor
    add_response = requests.post(
        f"{BASE_URL}/add_professor",
        json={
            "professor_name": professor_name,
            "number_of_articles": 3
        }
    )
    assert add_response.status_code == 200

    # 2. Search
    search_response = requests.post(
        f"{BASE_URL}/search",
        json={
            "professor_name": professor_name,
            "limit": 5
        }
    )
    assert search_response.status_code == 200

    # 3. Visualization
    viz_response = requests.post(
        f"{BASE_URL}/search_with_visualization",
        json={
            "professor_name": professor_name
        }
    )
    assert viz_response.status_code == 200

@pytest.mark.integration
class TestErrorScenarios:
    """Group of tests for error scenarios"""

    def test_nonexistent_professor_search(self):
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "professor_name": "NonexistentProfessor",
                "limit": 5
            }
        )
        assert response.status_code == 500

# Fixture for handling connection errors
@pytest.fixture
def check_api_available():
    try:
        requests.get(BASE_URL)
    except requests.ConnectionError:
        pytest.skip("API server is not available")

# Add this as a dependency to all tests
@pytest.mark.usefixtures("check_api_available")
class TestWithConnectionCheck:
    """Tests that require API connection"""
    pass