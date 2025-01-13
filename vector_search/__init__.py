from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from collections import Counter
import uuid
import atexit
import json
import time
import torch
import pickle
import os

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
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product similarity
        
        # Move index to GPU if available
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.professor_data = {}  # Store professor metadata
        self.load_index()
        
        # Register cleanup method
        atexit.register(self.cleanup)

    def load_index(self):
        """Load existing index and data if available"""
        index_path = f"{self.path}/{self.collection_name}.index"
        data_path = f"{self.path}/{self.collection_name}.pkl"
        
        if os.path.exists(index_path) and os.path.exists(data_path):
            self.index = faiss.read_index(index_path)
            with open(data_path, 'rb') as f:
                self.professor_data = pickle.load(f)

    def save_index(self):
        """Save index and data to disk"""
        os.makedirs(self.path, exist_ok=True)
        index_path = f"{self.path}/{self.collection_name}.index"
        data_path = f"{self.path}/{self.collection_name}.pkl"
        
        # If index is on GPU, convert back to CPU for saving
        if torch.cuda.is_available():
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
            
        with open(data_path, 'wb') as f:
            pickle.dump(self.professor_data, f)

    def cleanup(self):
        """Cleanup method"""
        try:
            self.save_index()
            print(f"\nIndex and data saved successfully.")
        except Exception as e:
            print(f"\nError during cleanup: {e}")

    def add_professor(self, 
                     name: str,
                     papers: List[Dict[str, str]],
                     department: str = None,
                     university: str = None):
        """Add a professor and their research papers to the database"""
        all_keywords = []
        all_summaries = []
        for paper in papers:
            all_keywords.extend([k.strip() for k in paper['Keywords'].split(',')])
            all_summaries.append(paper['summary'])

        keyword_freq = Counter(all_keywords)
        top_keywords = [k for k, _ in keyword_freq.most_common(20)]

        combined_text = f"{' '.join(top_keywords)} {' '.join(all_summaries)}"
        embedding = self.model.encode(combined_text)
        
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Store metadata
        professor_id = len(self.professor_data)
        self.professor_data[professor_id] = {
            'name': name,
            'department': department,
            'university': university,
            'papers': papers,
            'top_keywords': top_keywords,
            'keyword_frequencies': dict(keyword_freq),
            'paper_count': len(papers)
        }

    def find_similar_professors(self,
                              professor_name: str,
                              limit: int = 5,
                              min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """Find professors with similar research interests"""
        # Find professor's embedding
        professor_id = None
        for idx, data in self.professor_data.items():
            if data['name'] == professor_name:
                professor_id = idx
                break
                
        if professor_id is None:
            raise ValueError(f"Professor {professor_name} not found in database")

        # Get professor's vector
        professor_vector = None
        for i in range(self.index.ntotal):
            if i == professor_id:
                professor_vector = faiss.vector_to_array(self.index.reconstruct(i))
                break

        # Search similar professors
        D, I = self.index.search(np.array([professor_vector], dtype=np.float32), limit + 1)
        
        similar_profs = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx != professor_id and dist >= min_similarity:
                prof_data = self.professor_data[int(idx)]
                similar_profs.append({
                    'name': prof_data['name'],
                    'department': prof_data.get('department'),
                    'university': prof_data.get('university'),
                    'similarity_score': float(dist),
                    'shared_keywords': set(prof_data['top_keywords']) & 
                                     set(self.professor_data[professor_id]['top_keywords']),
                    'paper_count': prof_data['paper_count'],
                    'top_keywords': prof_data['top_keywords'][:5]
                })
        
        return similar_profs

    def get_professor_stats(self, professor_name: str) -> Dict[str, Any]:
        """Get statistics for a specific professor"""
        for prof_data in self.professor_data.values():
            if prof_data['name'] == professor_name:
                return {
                    'name': prof_data['name'],
                    'paper_count': prof_data['paper_count'],
                    'top_keywords': prof_data['top_keywords']
                }
        raise ValueError(f"Professor {professor_name} not found in database")

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
        time.sleep(1)
        with ProfessorResearchProfile(path="./professor_db") as profile_system:  
            profile_system.cleanup()
    except Exception as e:
        print(f"Error during database cleanup: {e}")
        raise