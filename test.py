from datetime import datetime
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import uuid
import atexit

class ProfessorResearchProfile:
    def __init__(self, 
                 collection_name: str = "professor_profiles",
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 location: str = ":memory:"):
        """
        Initialize the professor research profile system
        """
        self.client = QdrantClient(path=location)
        self.collection_name = collection_name
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Create collection
        self._create_collection()
        
        # Register cleanup method
        atexit.register(self.cleanup)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """
        Cleanup method to delete collection when program exits
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
            all_keywords.extend([k.strip() for k in paper['keywords'].split(',')])
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
                              min_similarity: float = 0.6) -> List[Dict[str, Any]]:
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

# Example professor data
professor_data = [
    {
        "name": "Dr. Sarah Chen",
        "papers": [
            {
                "title": "Advanced Deep Learning in Computer Vision",
                "keywords": "deep learning, CNN, image recognition, neural networks",
                "summary": "A comprehensive analysis of advanced deep learning techniques in modern computer vision applications."
            },
            {
                "title": "Transfer Learning in Medical Imaging",
                "keywords": "transfer learning, medical imaging, CNN, healthcare AI",
                "summary": "Investigation of transfer learning approaches for medical image analysis and diagnosis."
            },
            {
                "title": "Attention Mechanisms in Vision Transformers",
                "keywords": "transformers, attention mechanisms, computer vision, deep learning",
                "summary": "Study of attention mechanisms and their implementation in vision transformer architectures."
            }
        ]
    },
    # ... [Previous professor data remains the same] ...
]

if __name__ == "__main__":
    with ProfessorResearchProfile(location="./professor_db") as profile_system:
        # Add professors
        for prof_data in professor_data:
            profile_system.add_professor(
                name=prof_data["name"],
                papers=prof_data["papers"]
            )

        try:
            # Find similar professors
            similar_profs = profile_system.find_similar_professors(
                professor_name="Dr. Sarah Chen",
                limit=5
            )

            # Print results
            print("\nSimilar Professors to Dr. Sarah Chen:")
            for prof in similar_profs:
                print(f"\nName: {prof['name']}")
                print(f"Similarity Score: {prof['similarity_score']:.2f}")
                print(f"Shared Keywords: {', '.join(prof['shared_keywords'])}")
                print(f"Top Keywords: {', '.join(prof['top_keywords'])}")

            # Get professor statistics
            stats = profile_system.get_professor_stats("Dr. Sarah Chen")
            print(f"\nStatistics for {stats['name']}:")
            print(f"Papers: {stats['paper_count']}")
            print(f"Top Keywords: {', '.join(stats['top_keywords'])}")

        except Exception as e:
            print(f"An error occurred: {e}")