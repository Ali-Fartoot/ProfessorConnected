import pytest
from unittest.mock import patch, Mock, MagicMock
import os
import arxivpy
from crawler import crawl  

@pytest.fixture
def mock_arxiv():
    with patch('arxivpy.query') as mock_query, \
         patch('arxivpy.download') as mock_download, \
         patch('crawler.utils.convert_to_arxiv_query') as mock_convert:
        
        # Setup mock return values
        mock_convert.return_value = "Nathan Lambert"
        mock_article = MagicMock()
        mock_article.__getitem__.side_effect = lambda key: {
            "id": "1234.5678v9",
            "title": "Test Article: A Study in Testing"
        }.get(key)
        
        mock_query.return_value = [mock_article]
        
        # Simulate PDF creation when download is called
        def mock_download_effect(articles, path):
            os.makedirs(path, exist_ok=True)
            # Create a dummy PDF file
            pdf_path = os.path.join(path, "1234.5678v9.pdf")
            with open(pdf_path, 'w') as f:
                f.write("dummy pdf content")
        
        mock_download.side_effect = mock_download_effect
        
        yield {
            "query": mock_query,
            "download": mock_download,
            "convert": mock_convert
        }

def test_successful_crawl(mock_arxiv):
    # Arrange
    test_name = "Nathan Lambert"
    # Use project root directory instead of tmp_path
    expected_dir = os.path.join(os.getcwd(), "data", test_name)
    
    try:
        # Act
        crawl(test_name, number_of_articles=1)
        
        # Assert
        # Check if directory was created
        assert os.path.exists(expected_dir)
        assert os.path.isdir(expected_dir)
        
        # Check if PDF file exists in the directory
        pdf_files = [f for f in os.listdir(expected_dir) if f.endswith('.pdf')]
        assert len(pdf_files) == 1

        # Verify mock calls
        mock_arxiv['convert'].assert_called_once_with(test_name)
        mock_arxiv['query'].assert_called_once_with(
            search_query="Nathan Lambert",  # Updated to match mock_convert return value
            wait_time=3.0,
            sort_by='lastUpdatedDate'
        )
        mock_arxiv['download'].assert_called_once_with(
            mock_arxiv['query'].return_value[:1],
            path=str(expected_dir)
        )
    
    finally:
        # Cleanup: Remove the created directory after test
        if os.path.exists(expected_dir):
            for file in os.listdir(expected_dir):
                os.remove(os.path.join(expected_dir, file))
            os.rmdir(expected_dir)

def test_no_articles_found(mock_arxiv):
    # Arrange
    mock_arxiv['query'].return_value = []
    
    # Act/Assert
    with pytest.raises(ValueError) as exc_info:
        crawl("Test_Author")
    
    assert "No articles to download" in str(exc_info.value)
    mock_arxiv['download'].assert_not_called()