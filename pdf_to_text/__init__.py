from docling.document_converter import DocumentConverter
import yake
import os
import json
from .llm_extractor import SummarizerAgent, KeyextractorLLM
from keybert import KeyBERT

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
        self.key_extractor = KeyBERT()
        self.keywords_expander = KeyextractorLLM()

    def _section_chunker(self, text: str, symbol: str = "## ", 
                        sections=['Introduction', "Conclusion", "Discussion", "Future Works"]) -> list:
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
        """
        try:
            # Extract keywords from sections
            introduction_keywords = [i[0] for i in self.key_extractor.extract_keywords(sections[0])]
            conclusion_keywords = [i[0] for i in self.key_extractor.extract_keywords(sections[1])]
            
            # Get expanded keywords
            expanded_intro = self.keywords_expander.infer(sections[0])
            expanded_conclusion = self.keywords_expander.infer(sections[1])
            
            # Combine all keywords
            all_keywords = (
                introduction_keywords +
                conclusion_keywords +
                (expanded_intro if isinstance(expanded_intro, list) else [expanded_intro]) +
                (expanded_conclusion if isinstance(expanded_conclusion, list) else [expanded_conclusion])
            )
            
            # Remove duplicates and convert to string
            unique_keywords = list(set(filter(None, all_keywords)))
            
            return {
                "summaries": {
                    "sections": self.summarizer.infer("\n".join(sections))
                },
                "keywords": ", ".join(unique_keywords)
            }
        
        except Exception as e:
            print(f"Error in LLM processing: {e}")
            return {
                "summaries": "",
                "keywords": ""
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
            # Step 1: Convert the document to markdown
            result = self.document_converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()
            
            # Step 2: Extract sections from the markdown text
            selected_text = self._section_chunker(text=markdown_text)
            
            # Step 3: Extract keywords using traditional methods
            overall_keywords = self.keyword_extractor.extract_keywords(markdown_text)
            section_keywords = self.keyword_extractor.extract_keywords('\n'.join(selected_text))
            
            # Step 4: Perform LLM processing on the text
            llm_results = self._process_text_with_llm(markdown_text, selected_text)
            
            # Step 5: Combine all keywords into a unified string
            combined_keywords = (
                ", ".join([kw[0] for kw in overall_keywords]) + ", " +
                ", ".join([kw[0] for kw in section_keywords]) + ", " +
                llm_results["keywords"]
            )
            
            # Step 6: Return the results in a structured dictionary
            return {
                "summary": llm_results["summaries"],  # Extracted sections of the document
                "Keywords": combined_keywords,  # Combined keywords from all methods
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
                    "title": pdf.split(".pdf")[0],
                    **processed_data
                })

        # Save to JSON file
        output_path = os.path.join(author_path, f"{author_name}_analysis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(author_data, f, indent=4, ensure_ascii=False)