import pytest
from unittest.mock import Mock, patch
from pdf_to_text.llm_extractor import LLMAgent, SummarizerAgent, KeyExtractorLLM  # Replace with actual module name

@pytest.fixture
def mock_openai_client():
    with patch('openai.OpenAI') as mock:
        client = Mock()
        mock.return_value = client
        yield client

class TestKeyExtractorLLM:
    @pytest.fixture
    def key_extractor(self, mock_openai_client):
        return KeyExtractorLLM()

    def test_initialization(self, key_extractor):
        assert isinstance(key_extractor, LLMAgent)
        assert hasattr(key_extractor, 'key_extractor')

    def test_keyword_extraction_success(self, key_extractor):
        mock_keywords = [("KEYWORD1", 0.8), ("KEYWORD2", 0.7)]
        key_extractor.key_extractor.extract_keywords = Mock(return_value=mock_keywords)
        test_text = "Document about renewable energy and sustainability practices."
        result = key_extractor.infer(text=test_text)
        key_extractor.key_extractor.extract_keywords.assert_called_once_with(test_text)
        assert result == mock_keywords

    def test_keyword_format(self, key_extractor):
        test_text = "Website delivery issues and customer complaints."
        expected_keywords = ["E-COMMERCE", "DELIVERY MANAGEMENT", "CUSTOMER SATISFACTION"]
        key_extractor.key_extractor.extract_keywords = Mock(
            return_value=[(kw, 0.9) for kw in expected_keywords]
        )
        result = key_extractor.infer(text=test_text)
        extracted_keywords = [kw[0] for kw in result]
        
        assert all(kw.isupper() for kw in extracted_keywords)
        assert not any(kw in extracted_keywords for kw in ["WEBSITE", "CUSTOMER"])
        assert len(extracted_keywords) == len(set(extracted_keywords))  # No duplicates

    def test_empty_input_handling(self, key_extractor):
        key_extractor.key_extractor.extract_keywords = Mock(return_value=[])
        assert key_extractor.infer(text="") == []

def test_llm_agent_abc():
    with pytest.raises(TypeError):
        LLMAgent()