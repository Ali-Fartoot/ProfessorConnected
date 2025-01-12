from datetime import datetime
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import uuid

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
            pass

    def extract_paper_features(self, paper_text: str) -> Dict[str, str]:
        """
        Abstract method to extract features from a paper
        Override this method with your actual implementation
        """
        # This is a placeholder - implement your actual extraction logic
        pass

    def add_professor(self, 
                      name: str,
                      papers: List[Dict[str, str]],
                      department: str = None,
                      university: str = None):
        """
        Add a professor and their research papers to the database
        
        Args:
            name: Professor's name
            papers: List of dictionaries containing paper information:
                   {'title': str, 'keywords': str, 'summary': str}
            department: Professor's department (optional)
            university: Professor's university (optional)
        """
        # Combine all papers' keywords and summaries
        all_keywords = []
        all_summaries = []
        for paper in papers:
            all_keywords.extend([k.strip() for k in paper['keywords'].split(',')])
            all_summaries.append(paper['summary'])

        # Get frequency of keywords
        keyword_freq = Counter(all_keywords)
        top_keywords = [k for k, _ in keyword_freq.most_common(20)]

        # Create combined text for embedding
        combined_text = f"{' '.join(top_keywords)} {' '.join(all_summaries)}"
        embedding = self.model.encode(combined_text).tolist()

        # Create professor profile
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

        # Insert into database
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
      # First, get the professor's profile
      filter_query = models.Filter(
            must=[
                  models.FieldCondition(
                  key="name",
                  match=models.MatchText(text=professor_name)
                  )
            ]
      )
      
      # Get professor records
      prof_results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_query,
            limit=1,
            with_vectors=True  # Add this parameter to include vectors
      )[0]
      
      if not prof_results:
            raise ValueError(f"Professor {professor_name} not found in database")
      
      # Get the vector directly from the scroll results
      prof_vector = prof_results[0].vector
      
      if prof_vector is None:
            raise ValueError(f"Vector for professor {professor_name} not found")
      
      # Find similar professors
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
# Usage Example
if __name__ == "__main__":
    # Initialize the system
    profile_system = ProfessorResearchProfile(location="./professor_db")
    
    # Example: Adding professors with their papers
    prof1_papers = [
        {
            'title': 'Deep Learning Applications in Computer Vision',
            'keywords': 'deep learning, computer vision, CNN, object detection',
            'summary': 'A comprehensive study of deep learning applications in computer vision tasks.'
        },
        {
            'title': 'Neural Network Architectures',
            'keywords': 'neural networks, deep learning, architecture design',
            'summary': 'Analysis of various neural network architectures and their applications.'
        }
    ]
    
    prof2_papers = [
        {
            'title': 'Computer Vision for Autonomous Systems',
            'keywords': 'computer vision, robotics, autonomous systems',
            'summary': 'Exploring computer vision techniques in autonomous systems.'
        }
    ]
    
    # Add professors
    profile_system.add_professor(
        name="Dr. Smith",
        papers=prof1_papers,
        department="Computer Science",
        university="Tech University"
    )
    
    profile_system.add_professor(
        name="Dr. Johnson",
        papers=prof2_papers,
        department="Robotics",
        university="Innovation University"
    )
    
    # Find similar professors
    similar_profs = profile_system.find_similar_professors(
        professor_name="Dr. Smith",
        limit=5
    )
    
    # Print results
    print("\nSimilar Professors to Dr. Smith:")
    for prof in similar_profs:
        print(f"\nName: {prof['name']}")
        print(f"Department: {prof['department']}")
        print(f"University: {prof['university']}")
        print(f"Similarity Score: {prof['similarity_score']:.2f}")
        print(f"Shared Keywords: {', '.join(prof['shared_keywords'])}")
        print(f"Top Keywords: {', '.join(prof['top_keywords'])}")

    # Get professor statistics
    stats = profile_system.get_professor_stats("Dr. Smith")
    print(f"\nStatistics for {stats['name']}:")
    print(f"Papers: {stats['paper_count']}")
    print(f"Top Keywords: {', '.join(stats['top_keywords'])}")