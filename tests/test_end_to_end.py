import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, call
import os
import base64
from app import app 
print(app)
client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_dependencies():
    with patch('crawler.crawl') as mock_crawl, \
         patch('pdf_to_text.AuthorDocumentProcessor') as mock_processor, \
         patch('vector_search.add_professor') as mock_add_prof, \
         patch('vector_search.find_hybrid_search_professors') as mock_search, \
         patch('vector_search.visulizer.ProfessorVisualizer') as mock_viz, \
         patch('vector_search.cleanup_database') as mock_cleanup, \
         patch('os.makedirs') as mock_makedirs, \
         patch('os.path.exists') as mock_exists:
        
        # Configure mock processor
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_processor_instance.__call__ = MagicMock(return_value=True)
        
        # Configure visualization mock
        mock_viz_instance = MagicMock()
        mock_viz.return_value = mock_viz_instance
        mock_viz_instance.save_figures.return_value = "/tmp/network.png"
        
        # Configure exists mock with default behavior
        mock_exists.return_value = True
        
        # Configure makedirs to do nothing
        mock_makedirs.return_value = None
        
        yield {
            "crawl": mock_crawl,
            "processor": mock_processor_instance,
            "add_prof": mock_add_prof,
            "search": mock_search,
            "viz": mock_viz_instance,
            "cleanup": mock_cleanup,
            "exists": mock_exists,
            "makedirs": mock_makedirs
        }

def test_add_professor_success(mock_dependencies):
    professor_name = "Nathan Lambert"
    data_path = os.path.join("data", professor_name)
    json_path = os.path.join(data_path, f"{professor_name}.json")
    
    # Configure mock exists to return appropriate values for different paths
    mock_dependencies["exists"].side_effect = lambda path: {
        "data": True,
        data_path: True,
        json_path: True
    }.get(path, False)
    
    # Configure other mocks
    mock_dependencies["crawl"].return_value = True
    mock_dependencies["processor"].return_value.process_documents.return_value = True
    mock_dependencies["add_prof"].return_value = True
    
    response = client.post("/add_professor", json={
        "professor_name": professor_name,
        "number_of_articles": 3
    })
    
    # Print debug information
    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error response: {response.json()}")
    
    assert response.status_code == 200
    assert f"Successfully added professor {professor_name}" in response.json()["message"]
    
    # Verify the expected function calls
    mock_dependencies["crawl"].assert_called_once_with(professor_name, number_of_articles=3)
    mock_dependencies["add_prof"].assert_called_once_with(professor_name)
    

def test_add_professor_failure(mock_dependencies):
    professor_name = "Dr_Fail"
    data_path = os.path.join("data", professor_name)
    json_path = os.path.join(data_path, f"{professor_name}.json")

    # Configure mock exists to simulate a failure scenario
    # - data directory exists
    # - professor data directory exists
    # - but json file doesn't exist (processing failed)
    mock_dependencies["exists"].side_effect = lambda path: {
        "data": True,
        data_path: True,
        json_path: False  # This will trigger the failure case
    }.get(path, False)

    # Configure crawl to succeed but processing to fail
    mock_dependencies["crawl"].return_value = True
    mock_dependencies["processor"].return_value.__call__.side_effect = Exception("Failed to process documents")

    response = client.post("/add_professor", json={
        "professor_name": professor_name,
        "number_of_articles": 3
    })
    
    # Print debug information
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.json()}")

    assert response.status_code == 500
    assert "Failed to process professor documents" in response.json()["detail"]

    # Verify that crawl was called but add_professor was not
    mock_dependencies["crawl"].assert_called_once_with(professor_name, number_of_articles=3)
    mock_dependencies["add_prof"].assert_not_called()

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