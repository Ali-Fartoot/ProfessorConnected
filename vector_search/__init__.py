from datetime import datetime
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
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
        self.path = path  # Store path instead of location
        self.collection_name = collection_name
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.client = None  # Initialize client as None
        
        # Register cleanup method
        atexit.register(self.cleanup)
        
    def __enter__(self):
        """
        Context manager entry with retry mechanism
        """
        max_retries = 3
        retry_delay = 1
        last_exception = None

        for attempt in range(max_retries):
            try:
                self.client = QdrantClient(path=self.path)  # Use path instead of location
                self._create_collection()
                return self
            except RuntimeError as e:
                last_exception = e
                if "already accessed" in str(e) and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise last_exception
            except Exception as e:
                raise e
        
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit with proper cleanup
        """
        if self.client:
            self.client.close()
            self.client = None

        return False 
        
    def cleanup(self):
        """
        Cleanup method to delete collection
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"\nCollection {self.collection_name} has been deleted.")
        except Exception as e:
            print(f"\nError during cleanup: {e}")
        finally:
            self.client.close()
            
    def _create_collection(self):
        """Create collection for professor profiles"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                )
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

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                'name': name,
                'department': department,
                'university': university,
                'papers': papers,
                'top_keywords': top_keywords,
                'keyword_frequencies': dict(keyword_freq),
                'paper_count': len(papers),
            }
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def find_similar_professors(self,
                              professor_name: str,
                              limit: int = 5,
                              min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find professors with similar research interests
        """
        filter_query = models.Filter(
            must=[
                models.FieldCondition(
                    key="name",
                    match=models.MatchText(text=professor_name)
                )
            ]
        )
        
        prof_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_query,
            limit=1,
            with_vectors=True
        )[0]
        
        if not prof_results:
            raise ValueError(f"Professor {professor_name} not found in database")
        
        prof_vector = prof_results[0].vector
        
        if prof_vector is None:
            raise ValueError(f"Vector for professor {professor_name} not found")
        
        similar_profs = self.client.search(
            collection_name=self.collection_name,
            query_vector=prof_vector,
            query_filter=models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="name",
                        match=models.MatchText(text=professor_name)
                    )
                ]
            ),
            limit=limit,
            score_threshold=min_similarity
        )
        
        return [
            {
                'name': hit.payload['name'],
                'department': hit.payload.get('department'),
                'university': hit.payload.get('university'),
                'similarity_score': hit.score,
                'shared_keywords': set(hit.payload['top_keywords']) & 
                                 set(prof_results[0].payload['top_keywords']),
                'paper_count': hit.payload['paper_count'],
                'top_keywords': hit.payload['top_keywords'][:5]
            }
            for hit in similar_profs
        ]

    def get_professor_stats(self, professor_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific professor
        """
        filter_query = models.Filter(
            must=[
                models.FieldCondition(
                    key="name",
                    match=models.MatchText(text=professor_name)
                )
            ]
        )
        
        prof_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_query,
            limit=1
        )[0]
        
        if not prof_results:
            raise ValueError(f"Professor {professor_name} not found in database")
            
        prof_data = prof_results[0].payload
        return {
            'name': prof_data['name'],
            'paper_count': prof_data['paper_count'],
            'top_keywords': prof_data['top_keywords']
        }

# Add professor to professor_db
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

# Find similar professors
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
        time.sleep(1)
        with ProfessorResearchProfile(path="./professor_db") as profile_system:  
            profile_system.cleanup()
    except Exception as e:
        print(f"Error during database cleanup: {e}")
        raise