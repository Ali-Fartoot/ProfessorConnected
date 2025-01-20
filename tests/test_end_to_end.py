import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, call
import os
import base64
import app

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
        
        # Configure mock defaults
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
    # Test successful professor addition
    response = client.post("/add_professor", json={
        "professor_name": "Dr_Smith",
        "number_of_articles": 3
    })
    
    assert response.status_code == 200
    assert "Successfully added professor Dr_Smith" in response.json()["message"]
    
    # Verify pipeline execution
    mock_dependencies["crawl"].assert_called_once_with("Dr_Smith", number_of_articles=3)
    mock_dependencies["processor"].assert_called_once_with("Dr_Smith")
    mock_dependencies["add_prof"].assert_called_once_with("Dr_Smith")

def test_add_professor_failure(mock_dependencies):
    # Test processing failure
    mock_dependencies["exists"].side_effect = lambda x: True if "data/Dr_Fail" in x else False
    response = client.post("/add_professor", json={
        "professor_name": "Dr_Fail",
        "number_of_articles": 3
    })
    
    assert response.status_code == 500
    assert "Failed to process professor documents" in response.json()["detail"]

def test_search_visualization(mock_dependencies):
    # Test visualization endpoint
    mock_dependencies["search"].return_value = [
        {"name": "Dr_Smith", "similarity": 0.85},
        {"name": "Dr_Jones", "similarity": 0.75}
    ]
    
    response = client.post("/search_with_visualization", json={
        "professor_name": "Dr_Smith",
        "limit": 2,
        "min_similarity": 0.7
    })
    
    assert response.status_code == 200
    assert "network_image" in response.json()
    
    # Verify base64 image
    image_data = response.json()["network_image"]
    assert len(image_data) > 1000
    assert base64.b64decode(image_data[:100])  # Should not raise error
    
    # Verify visualization calls
    mock_dependencies["viz"].save_figures.assert_called_once_with(
        professor_name="Dr_Smith",
        output_dir="temp_visualizations",
        format="png",
        limit=2,
        min_similarity=0.7
    )

def test_text_search(mock_dependencies):
    # Test text search endpoint
    mock_results = [
        {"name": "Dr_Smith", "score": 0.92},
        {"name": "Dr_Jones", "score": 0.85}
    ]
    mock_dependencies["search"].return_value = mock_results
    
    response = client.post("/search", json={
        "professor_name": "Dr_Smith",
        "limit": 2,
        "min_similarity": 0.8
    })
    
    assert response.status_code == 200
    assert response.json()["results"] == mock_results
    mock_dependencies["search"].assert_called_once_with(
        professor_name="Dr_Smith",
        limit=2,
        min_similarity=0.8
    )

def test_database_cleanup(mock_dependencies):
    # Test database cleanup endpoint
    response = client.delete("/cleanup_database")
    
    assert response.status_code == 200
    assert "successfully cleaned up" in response.json()["message"].lower()
    mock_dependencies["cleanup"].assert_called_once()

def test_full_workflow(mock_dependencies):
    # Test complete user workflow
    # 1. Add professor
    client.post("/add_professor", json={
        "professor_name": "Dr_Workflow",
        "number_of_articles": 3
    })
    
    # 2. Search
    search_response = client.post("/search", json={
        "professor_name": "Dr_Workflow",
        "limit": 5
    })
    
    # 3. Visualize
    viz_response = client.post("/search_with_visualization", json={
        "professor_name": "Dr_Workflow"
    })
    
    # 4. Cleanup
    cleanup_response = client.delete("/cleanup_database")
    
    # Verify all steps executed
    assert search_response.status_code == 200
    assert viz_response.status_code == 200
    assert cleanup_response.status_code == 200
    mock_dependencies["cleanup"].assert_called_once()