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
            print(f"An Error Occured {e}")
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
      profile_system = ProfessorResearchProfile(location="./professor_db")
    

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
      {
            "name": "Dr. Michael Rodriguez",
            "papers": [
                  {
                  "title": "Natural Language Processing in Healthcare",
                  "keywords": "NLP, healthcare, machine learning, text mining",
                  "summary": "Applications of NLP techniques in processing medical records and healthcare documentation."
                  },
                  {
                  "title": "Sentiment Analysis Using BERT",
                  "keywords": "BERT, sentiment analysis, transformers, NLP",
                  "summary": "Implementation of BERT models for advanced sentiment analysis tasks."
                  }
            ]
      },
      {
            "name": "Dr. Emily Watson",
            "papers": [
                  {
                  "title": "Quantum Computing Algorithms",
                  "keywords": "quantum computing, algorithms, quantum optimization, quantum circuits",
                  "summary": "Development of novel algorithms for quantum computing applications."
                  },
                  {
                  "title": "Quantum Machine Learning",
                  "keywords": "quantum computing, machine learning, quantum algorithms, optimization",
                  "summary": "Integration of quantum computing principles with machine learning techniques."
                  },
                  {
                  "title": "Error Correction in Quantum Systems",
                  "keywords": "quantum error correction, quantum computing, fault tolerance",
                  "summary": "Analysis of error correction methods in quantum computing systems."
                  }
            ]
      },
      {
            "name": "Dr. James Kim",
            "papers": [
                  {
                  "title": "Cybersecurity in IoT Networks",
                  "keywords": "IoT, cybersecurity, network security, encryption",
                  "summary": "Analysis of security challenges and solutions in IoT networks."
                  },
                  {
                  "title": "Blockchain Security Protocols",
                  "keywords": "blockchain, security protocols, cryptography, distributed systems",
                  "summary": "Development of secure protocols for blockchain applications."
                  }
            ]
      },
      {
            "name": "Dr. Lisa Martinez",
            "papers": [
                  {
                  "title": "Cloud Computing Optimization",
                  "keywords": "cloud computing, optimization, distributed systems, scalability",
                  "summary": "Optimization techniques for cloud computing infrastructure and services."
                  },
                  {
                  "title": "Edge Computing Architecture",
                  "keywords": "edge computing, distributed systems, IoT, network architecture",
                  "summary": "Design and implementation of edge computing architectures."
                  },
                  {
                  "title": "Fog Computing Systems",
                  "keywords": "fog computing, distributed computing, IoT, edge computing",
                  "summary": "Analysis of fog computing systems and their applications."
                  }
            ]
      },
      {
            "name": "Dr. Robert Chang",
            "papers": [
                  {
                  "title": "Reinforcement Learning in Robotics",
                  "keywords": "reinforcement learning, robotics, AI, machine learning",
                  "summary": "Application of reinforcement learning algorithms in robotic systems."
                  },
                  {
                  "title": "Multi-Agent Learning Systems",
                  "keywords": "multi-agent systems, AI, machine learning, cooperation",
                  "summary": "Development of learning algorithms for multi-agent systems."
                  }
            ]
      },
      {
            "name": "Dr. Amanda Brooks",
            "papers": [
                  {
                  "title": "Big Data Analytics in Healthcare",
                  "keywords": "big data, healthcare analytics, data mining, machine learning",
                  "summary": "Analysis of big data applications in healthcare systems."
                  },
                  {
                  "title": "Predictive Analytics Models",
                  "keywords": "predictive analytics, machine learning, statistical modeling",
                  "summary": "Development of predictive analytics models for healthcare outcomes."
                  },
                  {
                  "title": "Data Mining in Electronic Health Records",
                  "keywords": "data mining, EHR, healthcare, pattern recognition",
                  "summary": "Application of data mining techniques to electronic health records."
                  }
            ]
      },
      {
            "name": "Dr. Thomas Wilson",
            "papers": [
                  {
                  "title": "Software Testing Automation",
                  "keywords": "software testing, automation, quality assurance, CI/CD",
                  "summary": "Development of automated testing frameworks for software systems."
                  },
                  {
                  "title": "DevOps Practices in Software Engineering",
                  "keywords": "DevOps, software engineering, continuous integration, automation",
                  "summary": "Analysis of DevOps practices and their implementation."
                  }
            ]
      },
      {
            "name": "Dr. Rachel Green",
            "papers": [
                  {
                  "title": "Human-Computer Interaction Design",
                  "keywords": "HCI, user interface, user experience, interaction design",
                  "summary": "Study of human-computer interaction principles and design patterns."
                  },
                  {
                  "title": "Accessibility in Mobile Applications",
                  "keywords": "accessibility, mobile computing, user interface, inclusive design",
                  "summary": "Research on accessibility features in mobile application design."
                  },
                  {
                  "title": "Virtual Reality Interface Design",
                  "keywords": "virtual reality, HCI, interface design, user experience",
                  "summary": "Design principles for virtual reality user interfaces."
                  }
            ]
      },
      {
            "name": "Dr. David Park",
            "papers": [
                  {
                  "title": "Network Security Protocols",
                  "keywords": "network security, cryptography, security protocols, cybersecurity",
                  "summary": "Analysis and development of network security protocols."
                  },
                  {
                  "title": "Intrusion Detection Systems",
                  "keywords": "intrusion detection, network security, machine learning, cybersecurity",
                  "summary": "Development of advanced intrusion detection systems."
                  },
                  {
                  "title": "Zero-Trust Security Architecture",
                  "keywords": "zero-trust, security architecture, network security, authentication",
                  "summary": "Implementation of zero-trust security principles in network systems."
                  }
            ]
      }
      ]



      for prof_data in professor_data:
            profile_system.add_professor(
                  name=prof_data["name"],
                  papers=prof_data["papers"]
            )

      similar_profs = profile_system.find_similar_professors(
                              professor_name="Dr. Sarah Chen",
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