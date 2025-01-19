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
                        sections={"start": ['Introduction'], "end": ["Conclusion", "Discussion", "Future Works", "Future Work"]}) -> list:
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
            # Find sections
            for part in sections.keys():
                find_part = False
                for section in sections[part]:
                    if section.lower() in chunk[:30].lower():
                        chunks.append(chunk)
                        find_part = True
                    
                    if find_part: break
            
            figure_matches = figure_pattern.finditer(chunk)
            for match in figure_matches:
                figure_text = match.group(1).strip()
                figures.append(figure_text)
                        

        
        assert len(chunks) == 2, f"Internal: Number of elements in returned chunks in parser isn't 2, {len(chunks)}"
        
        return chunks, figures

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
                [i[0] for i in self.keywords_expander.infer(" ".join(introduction_keywords))]+
                [i[0] for i in self.keywords_expander.infer(" ".join(conclusion_keywords))]+
                (expanded_intro if isinstance(expanded_intro, list) else [expanded_intro]) +
                (expanded_conclusion if isinstance(expanded_conclusion, list) else [expanded_conclusion])
            )
            
            # Flatten the list and remove None values
            flattened_keywords = []
            for item in all_keywords:
                if item is not None:
                    if isinstance(item, list):
                        flattened_keywords.extend(item)
                    else:
                        flattened_keywords.append(item)
            
            # Remove duplicates while preserving order
            unique_keywords = list(dict.fromkeys(flattened_keywords))
            
            return {
                "summaries": self.summarizer.infer("\n".join(sections)),
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
            selected_text, figures = self._section_chunker(text=markdown_text)
            
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
            
            figures_keywords_llm = self.keywords_expander.infer(" ".join(figures))
            filterd_keywords = self.keywords_expander.infer(figures_keywords_llm + combined_keywords)
            # Step 6: Return the results in a structured dictionary
            return {
                "summary": llm_results["summaries"],  
                "Keywords": combined_keywords,           
                "figures_llm": filterd_keywords,
            
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