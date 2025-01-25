import pytest
import requests
import os
from unittest.mock import patch, MagicMock

BASE_URL = "http://localhost:8000"

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    requests.delete(f"{BASE_URL}/cleanup_database")
    yield
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
            "number_of_articles": -1  
        }
    )
    
    assert response.status_code in [400, 500]  

@pytest.mark.integration
def test_search_functionality():
    professor_name = "Nathan Lambert"
    add_response = requests.post(
        f"{BASE_URL}/add_professor",
        json={
            "professor_name": professor_name,
            "number_of_articles": 3
        }
    )
    assert add_response.status_code == 200
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
    response = requests.delete(f"{BASE_URL}/cleanup_database")
    assert response.status_code == 200
    assert "successfully" in response.json()["message"].lower()

@pytest.mark.integration
def test_full_workflow():
    professor_name = "Nathan Lambert"
    add_response = requests.post(
        f"{BASE_URL}/add_professor",
        json={
            "professor_name": professor_name,
            "number_of_articles": 3
        }
    )
    assert add_response.status_code == 200

    search_response = requests.post(
        f"{BASE_URL}/search",
        json={
            "professor_name": professor_name,
            "limit": 5
        }
    )
    assert search_response.status_code == 200

    viz_response = requests.post(
        f"{BASE_URL}/search_with_visualization",
        json={
            "professor_name": professor_name
        }
    )
    assert viz_response.status_code == 200

@pytest.mark.integration
class TestErrorScenarios:
    def test_nonexistent_professor_search(self):
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "professor_name": "NonexistentProfessor",
                "limit": 5
            }
        )
        assert response.status_code == 500

@pytest.fixture
def check_api_available():
    try:
        requests.get(BASE_URL)
    except requests.ConnectionError:
        pytest.skip("API server is not available")

@pytest.mark.usefixtures("check_api_available")
class TestWithConnectionCheck:
    """Tests that require API connection"""
    pass