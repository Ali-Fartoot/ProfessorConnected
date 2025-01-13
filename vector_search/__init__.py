from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from collections import Counter
import uuid
import atexit
import json
import time

class ProfessorResearchProfile:
    def __init__(self, 
                 collection_name: str = "professor_profiles",
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 path: str = "./professor_db"):
        """
        Initialize the professor research profile system
        """
        self.path = path
        self.collection_name = collection_name
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = None
        
        # Register cleanup method
        atexit.register(self.cleanup)
        
    def __enter__(self):
        """
        Context manager entry
        """
        try:
            self._create_collection()
            return self
        except Exception as e:
            raise e
        
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit
        """
        return False 
        
    def cleanup(self):
        """
        Cleanup method to delete collection
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"\nCollection {self.collection_name} has been deleted.")
        except Exception as e:
            print(f"\nError during cleanup: {e}")
            
    def _create_collection(self):
        """Create collection for professor profiles"""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Collection creation notice: {e}")
            pass

    def add_professor(self, 
                     name: str,
                     papers: List[Dict[str, str]],
                     department: str = None,
                     university: str = None):
        """
        Add a professor and their research papers to the database
        """
        all_keywords = []
        all_summaries = []
        for paper in papers:
            all_keywords.extend([k.strip() for k in paper['Keywords'].split(',')])
            all_summaries.append(paper['summary'])

        keyword_freq = Counter(all_keywords)
        top_keywords = [k for k, _ in keyword_freq.most_common(20)]

        combined_text = f"{' '.join(top_keywords)} {' '.join(all_summaries)}"
        embedding = self.model.encode(combined_text).tolist()

        metadata = {
            'name': name,
            'department': department if department else "",
            'university': university if university else "",
            'top_keywords': json.dumps(top_keywords),
            'keyword_frequencies': json.dumps(dict(keyword_freq)),
            'paper_count': len(papers),
            'papers': json.dumps(papers)
        }

        self.collection.add(
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def find_similar_professors(self,
                              professor_name: str,
                              limit: int = 5,
                              min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find professors with similar research interests
        """
        # First, get the professor's data
        results = self.collection.get(
            where={"name": professor_name}
        )
        
        if not results['ids']:
            raise ValueError(f"Professor {professor_name} not found in database")
        
        prof_embedding = self.model.encode(results['documents'][0]).tolist()
        prof_metadata = results['metadatas'][0]
        
        # Query for similar professors
        similar_results = self.collection.query(
            query_embeddings=[prof_embedding],
            n_results=limit + 1,  # +1 because the professor themselves will be included
            where={"name": {"$ne": professor_name}}
        )
        
        similar_profs = []
        for i in range(len(similar_results['ids'][0])):
            metadata = similar_results['metadatas'][0][i]
            score = similar_results['distances'][0][i]
            
            if score > (1 - min_similarity):  # Convert cosine similarity threshold
                continue
                
            similar_profs.append({
                'name': metadata['name'],
                'department': metadata['department'],
                'university': metadata['university'],
                'similarity_score': 1 - score,  # Convert distance to similarity
                'shared_keywords': set(json.loads(metadata['top_keywords'])) & 
                                 set(json.loads(prof_metadata['top_keywords'])),
                'paper_count': metadata['paper_count'],
                'top_keywords': json.loads(metadata['top_keywords'])[:5]
            })
            
        return similar_profs

    def get_professor_stats(self, professor_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific professor
        """
        results = self.collection.get(
            where={"name": professor_name}
        )
        
        if not results['ids']:
            raise ValueError(f"Professor {professor_name} not found in database")
            
        metadata = results['metadatas'][0]
        return {
            'name': metadata['name'],
            'paper_count': metadata['paper_count'],
            'top_keywords': json.loads(metadata['top_keywords'])
        }

# The helper functions remain the same
def add_professor(name: str):
    try:
        with ProfessorResearchProfile(path="./professor_db") as profile_system:  
            with open(f'./data/{name}/{name}.json', 'r') as openfile:
                json_object = json.load(openfile)
                professor_name = list(json_object.keys())[0]
                profile_system.add_professor(
                    name=professor_name,
                    papers=json_object[professor_name]
                )
    except Exception as e:
        print("Error adding professor to database: ", e)
        raise

def find_similar_professor(limit: int = 5):
    try:
        with ProfessorResearchProfile(path="./professor_db") as profile_system:  
            similar_profs = profile_system.find_similar_professors(
                professor_name="Majid Nili Ahmadabadi",
                limit=limit
            )
        return similar_profs
    except Exception as e:
        print("Error finding similar professors: ", e)
        raise

def cleanup_database():
    try:
        with ProfessorResearchProfile(path="./professor_db") as profile_system:  
            profile_system.cleanup()
    except Exception as e:
        print(f"Error during database cleanup: {e}")
        raise