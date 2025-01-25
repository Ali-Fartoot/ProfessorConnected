import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pdf_to_text import AuthorDocumentProcessor
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
    with patch('pdf_to_text.llm_extractor.SummarizerAgent') as mock_summarizer, \
         patch('keybert.KeyBERT') as mock_keybert, \
         patch('pdf_to_text.llm_extractor.KeyExtractorLLM') as mock_key_expander:
        
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

def test_error_handling(tmp_path, mock_document_converter, mock_llm_agents):
    author_name = "error_author"
    author_dir = tmp_path / "data" / author_name
    author_dir.mkdir(parents=True)
    
    pdf_path = author_dir / "bad_paper.pdf"
    pdf_path.touch()
    
    mock_document_converter.convert.side_effect = Exception("Conversion error")
    
   