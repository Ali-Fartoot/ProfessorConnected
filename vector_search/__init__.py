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
        
    def cleanup(self, name):
        """
        Cleanup method to delete collection
        """
        try:
            # Check if collection exists before trying to delete it
            collections = self.client.list_collections()
            self.client.delete_collection(name=name)
            print(f"\nCollection {self.collection_name} has been deleted.")

        except Exception as e:
            print(f"\nError during cleanup: {e}")
            
    def _create_collection(self):
        """Create collection for professor profiles"""
        try:
            # Get existing collection or create new one
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection {self.collection_name} is ready.")
        except Exception as e:
            print(f"Collection creation error: {e}")
            raise

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
        all_titles = []
        for paper in papers:
            all_titles.append(paper["title"])
            all_keywords.extend([k.strip() for k in paper['Keywords'].split(',')])
            all_summaries.append(paper['summary'])

        keyword_freq = Counter(all_keywords)
        top_keywords = [k for k, _ in keyword_freq.most_common(20)]
        titles = ", ".join(all_titles)

        combined_text = f"{' '.join(top_keywords)} {' '.join(all_summaries)} {titles}"
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

        # Check if professor already exists and update if necessary
        existing_entries = self.collection.get(
            where={"name": name}
        )
        
        if existing_entries['ids']:
            # Update existing entry
            self.collection.update(
                ids=existing_entries['ids'][0],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[combined_text]
            )
            print(f"Updated existing profile for professor {name}")
        else:
            # Add new entry
            self.collection.add(
                documents=[combined_text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            print(f"Added new profile for professor {name}")

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
                print(f"Successfully processed professor {professor_name}")
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

def cleanup_database(name: str ="professor_profiles"):
    try:
        with ProfessorResearchProfile(path="./professor_db") as profile_system:  
            profile_system.cleanup(name)
    except Exception as e:
        print(f"Error during database cleanup: {e}")
        raise