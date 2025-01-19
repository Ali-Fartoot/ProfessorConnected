from docling.document_converter import DocumentConverter
import yake
import os
import json
from .llm_extractor import SummarizerAgent, KeyExtractorLLM
from keybert import KeyBERT
import re

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
        self.keywords_expander = KeyExtractorLLM()

    def _section_chunker(self, text: str, symbol: str = "## ", 
                        sections={"start": ['Introduction', "Abstract"], "end": ["Conclusion", "Discussion", "Future Works", "Future Work"]}) -> list:
        """
        Extract specific sections from the text and find figure captions with descriptions.
        
        Args:
            text (str): Text to process
            symbol (str): Section delimiter
            sections (dict): Dictionary of section names to extract
            
        Returns:
            tuple: (list of extracted section texts, list of figure captions with descriptions)
        """
        text_chunks = text.split(symbol)
        chunks = []
        figures = []
        
        # Regex pattern for finding figure captions
        figure_pattern = re.compile(r'(?i)(fig(?:ure)?\.?\s*\d+[.:]\s*.*?)(?:\n\n|\Z)', re.DOTALL)
        
        for chunk in text_chunks:
            # Flag to track if a section from "start" or "end" has been found
            found_start = False
            found_end = False
            
            # Check for sections in "start"
            for section in sections["start"]:
                if section.lower() in chunk[:30].lower():
                    chunks.append(chunk)
                    found_start = True
                    break  # Stop checking "start" sections once one is found
            
            # Check for sections in "end" only if no "start" section was found
            if not found_start:
                for section in sections["end"]:
                    if section.lower() in chunk[:30].lower():
                        chunks.append(chunk)
                        found_end = True
                        break  
            
            figure_matches = figure_pattern.finditer(chunk)
            for match in figure_matches:
                figure_text = match.group(1).strip()
                figures.append(figure_text)
                
        return chunks, figures
    
    def _process_text(self, figure: list,sections: list) -> dict:
        """
        Process text using LLM agents for summarization and keyword extraction.
        """
        try:
            # Extract keywords from sections
            papers_digest = " ".join(sections)
            figures_digest = ", ".join(figure) 
            traditional_keywords = [i[0] for i in self.key_extractor.extract_keywords(papers_digest)]
            llm_keywords = self.keywords_expander.infer(papers_digest)
            fugures_llm = self.keywords_expander.infer(figures_digest)

            if isinstance(traditional_keywords, list):
                traditional_keywords = " ".join(traditional_keywords)
            
            if isinstance(llm_keywords, list):
                llm_keywords = " ".join(llm_keywords)

            if isinstance(llm_keywords, list):
                figures_digest = " ".join(figures_digest)

            all_keywords  = figures_digest + llm_keywords + traditional_keywords
            unique_keywords = ", ".join(set(all_keywords ))

            
            return {
                "summaries": self.summarizer.infer(papers_digest),
                "keywords":unique_keywords
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
            result = self.document_converter.convert(file_path)
            markdown_text = result.document.export_to_markdown()
            selected_text, figures = self._section_chunker(text=markdown_text)
            llm_results = self._process_text(figures ,selected_text)

            return {
                "summary": llm_results["summaries"],  
                "keywords": llm_results["keywords"],           
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
        output_path = os.path.join(author_path, f"{author_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(author_data, f, indent=4, ensure_ascii=False)