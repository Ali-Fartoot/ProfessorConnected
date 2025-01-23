import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, call
import os
import base64
from app import app 

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch('crawler.crawl') as mock_crawl, \
         patch('pdf_to_text.AuthorDocumentProcessor') as mock_processor, \
         patch('vector_search.add_professor') as mock_add_prof, \
         patch('vector_search.find_hybrid_search_professors') as mock_search, \
         patch('vector_search.visulizer.ProfessorVisualizer') as mock_viz, \
         patch('vector_search.cleanup_database') as mock_cleanup, \
         patch('os.makedirs'), \
         patch('os.path.exists') as mock_exists:
        
        mock_exists.side_effect = lambda x: False
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_viz_instance = MagicMock()
        mock_viz.return_value = mock_viz_instance
        mock_viz_instance.save_figures.return_value = "/tmp/network.png"
        
        yield {
            "crawl": mock_crawl,
            "processor": mock_processor_instance,
            "add_prof": mock_add_prof,
            "search": mock_search,
            "viz": mock_viz_instance,
            "cleanup": mock_cleanup,
            "exists": mock_exists
        }

def test_add_professor_success(mock_dependencies):
    response = client.post("http://localhost:8000/add_professor", json={
        "professor_name": "Nathan Lambert",
        "number_of_articles": 3
    })
    
    assert response.status_code == 200
    assert "Successfully added professor Nathan Lambert" in response.json()["message"]
    
    mock_dependencies["crawl"].assert_called_once_with("Nathan Lambert", number_of_articles=3)
    mock_dependencies["processor"].assert_called_once_with("Nathan Lambert")
    mock_dependencies["add_prof"].assert_called_once_with("Nathan Lambert")

def test_add_professor_failure(mock_dependencies):
    mock_dependencies["exists"].side_effect = lambda x: True if "data/Dr_Fail" in x else False
    response = client.post("http://localhost:8000/add_professor", json={
        "professor_name": "Dr_Fail",
        "number_of_articles": 3
    })
    
    assert response.status_code == 500


def test_database_cleanup(mock_dependencies):
    response = client.delete("http://localhost:8000/cleanup_database")
    
    assert response.status_code == 200


def test_full_workflow(mock_dependencies):

    client.post("http://localhost:8000/add_professor", json={
        "professor_name": "Nathan Lambert",
    })
    
    search_response = client.post("http://localhost:8000/search", json={
        "professor_name": "Nathan Lambert",
        "limit": 5
    })
    
    viz_response = client.post("http://localhost:8000/search_with_visualization", json={
        "professor_name": "Nathan Lambert"
    })
    
    cleanup_response = client.delete("/http://localhost:8000/cleanup_database")
    assert search_response.status_code == 200
    assert viz_response.status_code == 200
    assert cleanup_response.status_code == 200