from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from collections import Counter
import uuid
import json
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
        

    def __enter__(self):
        try:
            self._create_collection()
            return self
        except Exception as e:
            raise e
        
    def __exit__(self, exc_type, exc_value, traceback):
        return False 
        
    def cleanup(self, name):
        try:
            collections = self.client.list_collections()
            self.client.delete_collection(name=name)
            print(f"\nCollection {self.collection_name} has been deleted.")

        except Exception as e:
            print(f"\nError during cleanup: {e}")
            
    def _create_collection(self):
        try:
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
        all_keywords = []
        all_summaries = []
        all_titles = []
        for paper in papers:
            all_titles.append(paper["title"])
            paper_keywords = [k.strip().lower() for k in paper['keywords'].split(',')]
            all_keywords.extend(paper_keywords)
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
            'keywords': json.dumps(list(set(all_keywords))),
            'keyword_frequencies': json.dumps(dict(keyword_freq)),
            'paper_count': len(papers),
            'papers': json.dumps(papers)
        }
        existing_entries = self.collection.get(
            where={"name": name}
        )
        
        if existing_entries['ids']:
            self.collection.update(
                ids=existing_entries['ids'][0],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[combined_text]
            )
            print(f"Updated existing profile for professor {name}")
        else:
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

        results = self.collection.get(
            where={"name": professor_name}
        )
        if not results['ids']:
            raise ValueError(f"Professor {professor_name} not found in database")
        
        prof_embedding = self.model.encode(results['documents'][0]).tolist()
        prof_metadata = results['metadatas'][0]

        similar_results = self.collection.query(
            query_embeddings=[prof_embedding],
            n_results=limit + 1, 
            where={"name": {"$ne": professor_name}}
        )
        
        similar_profs = []
        for i in range(len(similar_results['ids'][0])):
            metadata = similar_results['metadatas'][0][i]
            score = similar_results['distances'][0][i]
            
            if score > (1 - min_similarity):
                continue
                
            similar_profs.append({
                'name': metadata['name'],
                'department': metadata['department'],
                'university': metadata['university'],
                'similarity_score': 1 - score,
                'shared_keywords': set(json.loads(metadata['top_keywords'])) & 
                                 set(json.loads(prof_metadata['top_keywords'])),
                'paper_count': metadata['paper_count'],
                'top_keywords': json.loads(metadata['top_keywords'])[:5]
            })
            
        return similar_profs

    def get_professor_stats(self, professor_name: str) -> Dict[str, Any]:
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
    
    def hybrid_search_by_professor(self, 
                                professor_name: str,
                                limit: int = 5,
                                min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        try:
            prof_results = self.collection.get(
                where={"name": professor_name}
            )
            
            if not prof_results['ids']:
                raise ValueError(f"Professor {professor_name} not found in database")
                
            prof_metadata = prof_results['metadatas'][0]
            prof_document = prof_results['documents'][0]
            
            prof_embedding = self.model.encode(prof_document).tolist()
            
            prof_keywords = json.loads(prof_metadata['keywords'])
            prof_top_keywords = json.loads(prof_metadata['top_keywords'])
            
            similar_results = self.collection.query(
                query_embeddings=[prof_embedding],
                n_results=limit * 2,
                where={"name": {"$ne": professor_name}}
            )
            
            results = {}
            for i in range(len(similar_results['ids'][0])):
                metadata = similar_results['metadatas'][0][i]
                similarity_score = 1 - similar_results['distances'][0][i]
                
                if similarity_score < min_similarity:
                    continue
                    
                similar_prof_name = metadata['name']
                similar_prof_keywords = set(json.loads(metadata['keywords']))
                
                keyword_overlap = len(set(prof_keywords) & similar_prof_keywords)
                keyword_score = keyword_overlap / len(prof_keywords) if prof_keywords else 0
                
                shared_top_keywords = set(prof_top_keywords) & set(json.loads(metadata['top_keywords']))
                results[similar_prof_name] = {
                    'name': similar_prof_name,
                    'embedding_similarity': similarity_score,
                    'keyword_similarity': keyword_score,
                    'combined_score': (similarity_score + keyword_score) / 2,
                    'paper_count': metadata['paper_count'],
                    'shared_top_keywords': list(shared_top_keywords),
                    'top_keywords': json.loads(metadata['top_keywords'])[:5],
                    'papers': json.loads(metadata['papers'])
                }
            
            final_results = list(results.values())
            final_results.sort(key=lambda x: x['combined_score'], reverse=True)
            return final_results[:limit]
            
        except Exception as e:
            print(f"Error in hybrid search by professor: {e}")
            raise

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


        
def cleanup_database(name: str = "professor_profiles"):
    try:
        with ProfessorResearchProfile(path="./professor_db") as profile_system:  
            profile_system.cleanup(name)
    except Exception as e:
        print(f"Error during database cleanup: {e}")
        raise

def find_hybrid_search_professors(professor_name: str,  limit: int = 5, min_similarity: float = 0.3):
    try:
        with ProfessorResearchProfile(path="./professor_db") as profile_system:
            results = profile_system.hybrid_search_by_professor(
                professor_name=professor_name,
                limit=limit,
                min_similarity=min_similarity
            )
            return results
    except Exception as e:
        print("Error in hybrid search:", e)
        raise