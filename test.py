from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import uuid

class LocalHybridSearch:
    def __init__(self, 
                 collection_name: str,
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 location: str = ":memory:"):  # Use ":memory:" for in-memory or path for persistence
        """
        Initialize hybrid search with local vector database
        
        Args:
            collection_name: Name of the collection to store vectors
            embedding_model: Name of the sentence-transformer model to use
            location: ":memory:" for in-memory DB or path to store the database
        """
        # Initialize Qdrant client (locally, no server needed)
        self.client = QdrantClient(path=location)
        self.collection_name = collection_name
        
        # Initialize the embedding model
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Create collection if it doesn't exist
        self._create_collection()
        
    def _create_collection(self):
        """Create a new collection with necessary configuration"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim,
                    distance=models.Distance.COSINE
                )
            )
        except Exception as e:
            # Collection might already exist
            pass

    def insert_documents(self, documents: List[Dict[str, str]]):
        """
        Insert documents with their keywords and summaries
        
        Args:
            documents: List of dictionaries containing:
                - 'text': original text
                - 'keywords': keywords
                - 'summary': text summary
        """
        points = []
        
        for doc in documents:
            # Generate embedding from combined text
            combined_text = f"{doc['summary']} {doc['keywords']}"
            embedding = self.model.encode(combined_text).tolist()
            
            # Create point
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'keywords': doc['keywords'],
                    'summary': doc['summary'],
                    'text': doc.get('text', ''),  # Original text is optional
                    # Create keyword list for better filtering
                    'keyword_list': [k.strip() for k in doc['keywords'].split(',')]
                }
            ))
        
        # Insert in batch
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def hybrid_search(self,
                     query_text: str,
                     keyword_filter: Optional[str] = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and keyword filtering
        
        Args:
            query_text: Search query text
            keyword_filter: Optional keyword to filter results
            limit: Maximum number of results to return
        """
        # Generate query embedding
        query_vector = self.model.encode(query_text).tolist()
        
        # Prepare filter
        filter_query = None
        if keyword_filter:
            filter_query = models.Filter(
                must=[
                    models.FieldCondition(
                        key="keyword_list",
                        match=models.MatchText(
                            text=keyword_filter
                        )
                    )
                ]
            )
        
        # Perform search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=filter_query,
            limit=limit
        )
        
        # Format results
        results = []
        for hit in search_results:
            results.append({
                'id': hit.id,
                'score': hit.score,
                'keywords': hit.payload['keywords'],
                'summary': hit.payload['summary'],
                'text': hit.payload.get('text', '')
            })
            
        return results

    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(self.collection_name)

# Advanced features class that inherits from LocalHybridSearch
class AdvancedHybridSearch(LocalHybridSearch):
    def semantic_keyword_search(self, 
                              query_text: str, 
                              min_score: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform semantic search with keyword boosting
        """
        query_vector = self.model.encode(query_text).tolist()
        
        # Search with scoring
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=20,  # Get more results initially for filtering
            score_threshold=min_score
        )
        
        return [
            {
                'id': hit.id,
                'score': hit.score,
                'keywords': hit.payload['keywords'],
                'summary': hit.payload['summary']
            }
            for hit in results
        ]

    def keyword_faceted_search(self, 
                             keywords: List[str], 
                             limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform faceted search based on keywords
        """
        results = {}
        
        for keyword in keywords:
            filter_query = models.Filter(
                must=[
                    models.FieldCondition(
                        key="keyword_list",
                        match=models.MatchText(text=keyword)
                    )
                ]
            )
            
            hits = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_query,
                limit=limit
            )[0]
            
            results[keyword] = [
                {
                    'id': hit.id,
                    'keywords': hit.payload['keywords'],
                    'summary': hit.payload['summary']
                }
                for hit in hits
            ]
            
        return results

# Usage Example
if __name__ == "__main__":
    # Initialize with in-memory storage
    search_engine = LocalHybridSearch(
        collection_name="documents",
        location=":memory:"  # For persistence, use a path like "./local_db"
    )
    
    # Example documents
    documents = [
        {
            'text': 'Comprehensive guide to machine learning algorithms...',
            'keywords': 'machine learning, AI, algorithms, neural networks',
            'summary': 'An in-depth overview of various machine learning algorithms and their applications.'
        },
        {
            'text': 'Data visualization techniques for big data...',
            'keywords': 'data visualization, matplotlib, plotting, analytics',
            'summary': 'Exploring different techniques for visualizing large datasets effectively.'
        }
    ]
    
    # Insert documents
    search_engine.insert_documents(documents)
    
    # Basic hybrid search
    results = search_engine.hybrid_search(
        query_text="machine learning visualization",
        keyword_filter="visualization",
        limit=5
    )
    
    # Print results
    for result in results:
        print(f"\nScore: {result['score']:.4f}")
        print(f"Keywords: {result['keywords']}")
        print(f"Summary: {result['summary']}")
    
    # Advanced usage
    advanced_search = AdvancedHybridSearch(
        collection_name="advanced_documents",
        location=":memory:"
    )
    
    # Semantic keyword search
    semantic_results = advanced_search.semantic_keyword_search(
        query_text="deep learning applications",
        min_score=0.7
    )
    
    # Faceted search by keywords
    faceted_results = advanced_search.keyword_faceted_search(
        keywords=["machine learning", "visualization"],
        limit=5
    )