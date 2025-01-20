import pytest
from unittest.mock import patch, Mock, MagicMock
import os
import arxivpy
from crawler import crawl  

@pytest.fixture
def mock_arxiv():
    with patch('arxivpy.query') as mock_query, \
         patch('arxivpy.download') as mock_download, \
         patch('your_module.convert_to_arxiv_query') as mock_convert:
        
        # Setup mock return values
        mock_convert.return_value = "test_query"
        mock_article = MagicMock()
        mock_article.__getitem__.side_effect = lambda key: {
            "id": "1234.5678v9",
            "title": "Test Article: A Study in Testing"
        }.get(key)
        
        mock_query.return_value = [mock_article]
        mock_download.return_value = None
        
        yield {
            "query": mock_query,
            "download": mock_download,
            "convert": mock_convert
        }

def test_successful_crawl(tmp_path, mock_arxiv):
    # Arrange
    test_name = "Dr_Test_Author"
    expected_dir = tmp_path / "data" / test_name
    expected_sanitized = "Test_Article_A_Study_in_Testing.pdf"
    
    # Act
    crawl(test_name, number_of_articles=1)
    
    # Assert
    mock_arxiv['convert'].assert_called_once_with(test_name)
    mock_arxiv['query'].assert_called_once_with(
        search_query="test_query",
        wait_time=3.0,
        sort_by='lastUpdatedDate'
    )
    mock_arxiv['download'].assert_called_once_with(
        mock_arxiv['query'].return_value[:1],
        path=str(expected_dir)
    )
    
    # Verify file renaming logic
    expected_old_path = expected_dir / "1234.5678v9.pdf"
    expected_new_path = expected_dir / expected_sanitized
    assert not os.path.exists(expected_old_path)  # Old filename shouldn't exist
    assert os.path.exists(expected_new_path)  # New filename should exist

def test_no_articles_found(mock_arxiv):
    # Arrange
    mock_arxiv['query'].return_value = []
    
    # Act/Assert
    with pytest.raises(ValueError) as exc_info:
        crawl("Test_Author")
    
    assert "No articles to download" in str(exc_info.value)
    mock_arxiv['download'].assert_not_called()

def test_filename_sanitization(tmp_path, mock_arxiv):
    # Arrange
    test_name = "Test_Author"
    mock_article = MagicMock()
    mock_article.__getitem__.side_effect = lambda key: {
        "id": "0000.0000v0",
        "title": "Invalid/File:Name*With?Special<>Characters"
    }.get(key)
    mock_arxiv['query'].return_value = [mock_article]
    
    # Act
    crawl(test_name, number_of_articles=1)
    
    # Assert
    expected_sanitized = "Invalid_File_Name_With_Special_Characters.pdf"
    expected_path = tmp_path / "data" / test_name / expected_sanitized
    assert os.path.exists(expected_path)

def test_missing_file_handling(tmp_path, mock_arxiv, capsys):
    # Arrange
    test_name = "Test_Author"
    mock_article = MagicMock()
    mock_article.__getitem__.side_effect = lambda key: {
        "id": "missing_id",
        "title": "Missing File Test"
    }.get(key)
    mock_arxiv['query'].return_value = [mock_article]
    
    # Act
    crawl(test_name, number_of_articles=1)
    
    # Assert
    captured = capsys.readouterr()
    assert "File not found" in captured.out
    assert not os.path.exists(tmp_path / "data" / test_name / "Missing_File_Test.pdf")