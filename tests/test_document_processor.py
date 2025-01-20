import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pdf_to_text import AuthorDocumentProcessor  # Replace with actual module name
from keybert import KeyBERT

@pytest.fixture
def mock_document_converter():
    mock = Mock()
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = """
        ## Introduction
        This is a sample introduction about artificial intelligence and machine learning.
        
        ## Methodology
        We propose a novel deep learning approach using convolutional neural networks.
        
        Figure 1: Architecture diagram of the proposed model
    """
    mock.convert.return_value = mock_result
    return mock

@pytest.fixture
def mock_llm_agents():
    with patch('llm.SummarizerAgent') as mock_summarizer, \
         patch('KeyBERT') as mock_keybert, \
         patch('llm_extractor.KeyExtractorLLM') as mock_key_expander:
        
        mock_summarizer_instance = Mock()
        mock_summarizer_instance.infer.return_value = "Sample summary about AI and ML techniques"
        
        mock_keybert_instance = Mock()
        mock_keybert_instance.extract_keywords.return_value = [("AI", 0.8), ("ML", 0.7)]
        
        mock_key_expander_instance = Mock()
        mock_key_expander_instance.infer.side_effect = [
            ["ARTIFICIAL_INTELLIGENCE", "MACHINE_LEARNING"],
            ["NEURAL_NETWORKS", "DEEP_LEARNING"],
            ["AI, ML, DEEP LEARNING"]
        ]
        
        mock_summarizer.return_value = mock_summarizer_instance
        mock_keybert.return_value = mock_keybert_instance
        mock_key_expander.return_value = mock_key_expander_instance
        
        yield {
            "summarizer": mock_summarizer_instance,
            "keybert": mock_keybert_instance,
            "key_expander": mock_key_expander_instance
        }

def test_full_processing_flow(tmp_path, mock_document_converter, mock_llm_agents):
    # Setup test environment
    author_name = "test_author"
    author_dir = tmp_path / "data" / author_name
    author_dir.mkdir(parents=True)
    
    # Create dummy PDF file
    pdf_path = author_dir / "test_paper.pdf"
    pdf_path.write_bytes(b"dummy pdf content")
    
    # Initialize processor with test paths
    processor = AuthorDocumentProcessor(base_data_path=str(tmp_path / "data"))
    processor.document_converter = mock_document_converter
    
    # Execute processing
    processor(author_name)
    
    # Verify output file
    output_file = author_dir / f"{author_name}.json"
    assert output_file.exists()
    
    # Verify JSON content
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert author_name in data
    assert len(data[author_name]) == 1
    paper_data = data[author_name][0]
    
    # Validate structure
    assert "title" in paper_data
    assert "summary" in paper_data
    assert "keywords" in paper_data
    
    # Validate content
    assert paper_data["title"] == "test_paper"
    assert paper_data["summary"] == "Sample summary about AI and ML techniques"
    assert "DEEP LEARNING" in paper_data["keywords"]
    
    # Verify mock calls
    mock_document_converter.convert.assert_called_once_with(str(pdf_path))
    mock_llm_agents["summarizer"].infer.assert_called_once()
    mock_llm_agents["keybert"].extract_keywords.assert_called_once()
    
    # Verify keyword expansion calls
    assert mock_llm_agents["key_expander"].infer.call_count == 3
    assert "AI, ML" in mock_llm_agents["key_expander"].infer.call_args_list[2][0][0]

def test_error_handling(tmp_path, mock_document_converter, mock_llm_agents):
    # Test error handling in PDF processing
    author_name = "error_author"
    author_dir = tmp_path / "data" / author_name
    author_dir.mkdir(parents=True)
    
    # Create empty PDF file
    pdf_path = author_dir / "bad_paper.pdf"
    pdf_path.touch()
    
    # Force conversion error
    mock_document_converter.convert.side_effect = Exception("Conversion error")
    
   