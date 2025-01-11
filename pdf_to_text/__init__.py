from docling.document_converter import DocumentConverter
import yake
import os
import json
from .llm_extractor import SummarizerAgent, KeyExtractorAgent

class AuthorDocumentProcessor:
    def __init__(self, base_data_path='data'):
        """
        Initialize the AuthorDocumentProcessor.
        
        Args:
            base_data_path (str): Base path where author data is stored
        """
        self.base_data_path = base_data_path
        self.document_converter = DocumentConverter()
        self.keyword_extractor = yake.KeywordExtractor()
        
        # Initialize LLM agents
        self.summarizer = SummarizerAgent()
        self.key_extractor = KeyExtractorAgent()

    def _section_chunker(self, text: str, symbol: str = "## ", 
                        sections=['Introduction', "Conclusion"]) -> list:
        """
        Extract specific sections from the text.
        
        Args:
            text (str): Text to process
            symbol (str): Section delimiter
            sections (list): List of section names to extract
            
        Returns:
            list: List of extracted section texts
        """
        text_chunks = text.split(symbol)
        chunks = []
        for chunk in text_chunks:
            for section in sections:
                if section.lower() in chunk[:15].lower():
                    chunks.append(chunk)
        return chunks

    def _process_text_with_llm(self, text: str, sections: list) -> dict:
        """
        Process text using LLM agents for summarization and keyword extraction.
        
        Args:
            text (str): Full text to process
            sections (list): List of extracted sections
            
        Returns:
            dict: Dictionary containing LLM processing results
        """
        try:

            print("\n".join(sections))
            print("-------------------------------")
            print(text)
            print("-------------------------------")
            print(sections)
            xd
            # llm_results = {
            #     "summaries": {
            #         "overall": self.summarizer.infer(text),
            #         "sections": self.summarizer.infer("\n".join(sections))
            #     },
            #     "keywords": {
            #         "overall": self.key_extractor.infer(text),
            #         "introduction": self.key_extractor.infer(sections[0]) if sections else [],
            #         "conclusion": self.key_extractor.infer(sections[1]) if len(sections) > 1 else []
            #     }
            #}
            return None
        except Exception as e:
            print(f"Error in LLM processing: {str(e)}")
            return {
                "summaries": {"overall": "", "sections": ""},
                "keywords": {"overall": [], "introduction": [], "conclusion": []}
            }

    def _pdf_to_json(self, file_path: str) -> dict:
        """
        Convert PDF to structured data.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            dict: Structured data from PDF including traditional and LLM-based analysis
        """
        try:
            # Convert PDF to markdown
            result = self.document_converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()
            
            # Extract sections
            selected_text = self._section_chunker(text=markdown_text)
            
            # Traditional keyword extraction
            overall_keywords = self.keyword_extractor.extract_keywords(markdown_text)
            section_keywords = self.keyword_extractor.extract_keywords('\n'.join(selected_text))
            
            # LLM processing
            llm_results = self._process_text_with_llm(markdown_text, selected_text)
            
            return {
                "text": selected_text,
                "traditional_analysis": {
                    "overall_keywords": [kw[0] for kw in overall_keywords],
                    "section_keywords": [kw[0] for kw in section_keywords]
                },
                "llm_analysis": llm_results
            }
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
            return None

    def __call__(self, author_name: str) -> None:
        """
        Process all PDFs for an author and save results to JSON.
        
        Args:
            author_name (str): Name of the author/professor
        """
        # Setup paths
        author_path = os.path.join(self.base_data_path, author_name)
        author_pdfs = [f for f in os.listdir(author_path) if f.endswith('.pdf')]
        
        # Initialize result dictionary
        author_data = {author_name: []}

        # Process each PDF
        for pdf in author_pdfs:
            pdf_path = os.path.join(author_path, pdf)
            processed_data = self._pdf_to_json(pdf_path)
            
            if processed_data:
                author_data[author_name].append({
                    "title": pdf,
                    **processed_data
                })

        # Save to JSON file
        output_path = os.path.join(author_path, f"{author_name}_analysis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(author_data, f, indent=4, ensure_ascii=False)