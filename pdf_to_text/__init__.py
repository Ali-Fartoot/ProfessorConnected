from docling.document_converter import DocumentConverter
import yake
import os
import json

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

    def _pdf_to_json(self, file_path: str) -> tuple:
        """
        Convert PDF to structured data.
        
        Args:
            file_path (str): Path to PDF file
            
        Returns:
            tuple: (selected_text, overall_keywords, section_keywords)
        """
        # Convert PDF to markdown
        result = self.document_converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        
        # Extract keywords
        overall_keywords = self.keyword_extractor.extract_keywords(markdown_text)
        selected_text = self._section_chunker(text=markdown_text)
        section_keywords = self.keyword_extractor.extract_keywords('\n'.join(selected_text))

        return (
            selected_text,
            [kw[0] for kw in overall_keywords],
            [kw[0] for kw in section_keywords]
        )

    def __call__(self, author_name: str) -> None:
        """
        Process all PDFs for an author and save results to JSON.
        
        Args:
            author_name (str): Name of the author/professor
        """
        # Setup paths
        author_path = os.path.join(self.base_data_path, author_name)
        author_pdfs = os.listdir(author_path)
        
        # Initialize result dictionary
        author_data = {author_name: []}

        # Process each PDF
        for pdf in author_pdfs:
            pdf_path = os.path.join(author_path, pdf)
            section_text, overall_kw, section_kw = self._pdf_to_json(pdf_path)
            
            # Add processed data to result
            author_data[author_name].append({
                "title": pdf, 
                'text': section_text,
                'overall_keywords': overall_kw,
                'section_keywords': section_kw
            })

        # Save to JSON file
        output_path = os.path.join(author_path, f"{author_name}.json")
        with open(output_path, 'w') as f:
            json.dump(author_data, f, indent=4)


