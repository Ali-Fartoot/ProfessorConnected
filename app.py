from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from crawler import crawl
from pdf_to_text import AuthorDocumentProcessor
from vector_search import cleanup_database, add_professor, find_hybrid_search_professors
from vector_search.visulizer import ProfessorVisualizer
import base64
import io

app = FastAPI(title="Professor Research Profile API")

class ProfessorRequest(BaseModel):
    name: str
    number_of_articles: int = 3

class SearchRequest(BaseModel):
    text_query: str
    keywords: Optional[List[str]] = None
    limit: int = 5
    weight_embedding: float = 0.6
    threshold: float = 0.3

@app.post("/add_professor")
async def add_new_professor(request: ProfessorRequest):
    """
    Add a new professor to the database by crawling and processing their papers
    """
    try:
        name = request.name
        data_path = os.path.join("data", name)
        json_path = os.path.join(data_path, f"{name}.json")

        os.makedirs("data", exist_ok=True)

        # Step 1: Crawl papers
        if not os.path.exists(data_path):
            crawl(name, number_of_articles=request.number_of_articles)
        
        # Step 2: Process documents
        if os.path.exists(data_path) and not os.path.exists(json_path):
            document_processor = AuthorDocumentProcessor()
            document_processor(name)

        # Step 3: Add professor to database
        if os.path.exists(json_path):
            add_professor(name)
            return {"message": f"Successfully added professor {name} to database"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process professor documents")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_with_visualization")
async def search_and_visualize(request: SearchRequest):
    """
    Search for similar professors and return visualization images
    """
    try:
        visualizer = ProfessorVisualizer()
        
        # Create temporary directory for images
        output_dir = "temp_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        network_path = visualizer.save_figures(
            professor_name=request.professor_name,
            output_dir=output_dir,
            format="png",
            limit=request.limit,
            min_similarity=request.min_similarity
    
        )
    
        with open(network_path, "rb") as f:
            network_image = base64.b64encode(f.read()).decode()
        

        
        # Cleanup temporary files
        os.remove(network_path)
        os.rmdir(output_dir)
        
        return {
            "network_image": network_image,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_professors(request: SearchRequest):
    """
    Search for similar professors and return results as text
    """
    try:
        results = find_hybrid_search_professors(
            professor_name=request.professor_name, 
            limit=request.limit,
            min_similarity=request.min_similarity

        )
        
        return {
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup_database")
async def cleanup():
    """
    Remove all data from the database
    """
    try:
        cleanup_database()
        return {"message": "Database successfully cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

