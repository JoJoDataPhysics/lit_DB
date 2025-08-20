import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime
import json

from src.models import AnalysisResult


class VectorDatabaseManager:
    """Simplified vector database manager with fallback implementation."""
    
    def __init__(self, config):
        self.config = config.vector_db
        self.logger = logging.getLogger(__name__)
        
        # Try to load dependencies
        try:
            self._initialize_dependencies()
            self.dependencies_available = True
        except ImportError as e:
            self.logger.warning(f"Vector database dependencies not fully available: {e}")
            self.dependencies_available = False
            # Use fallback mode
            self._init_fallback_mode()
    
    def _initialize_dependencies(self):
        """Initialize vector database dependencies."""
        try:
            import numpy as np
            import chromadb
            from sklearn.cluster import KMeans
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Store imports
            self.np = np
            self.chromadb = chromadb
            self.KMeans = KMeans
            self.cosine_similarity = cosine_similarity
            
            # Set this early so other methods can check it
            self.dependencies_available = True
            
            # Try to initialize sentence transformers
            try:
                from sentence_transformers import SentenceTransformer
                self.SentenceTransformer = SentenceTransformer
                self.use_real_embeddings = True
                self.logger.info("Using real sentence transformers for embeddings")
            except ImportError:
                self.use_real_embeddings = False
                self.logger.warning("sentence-transformers not available, using hash-based embeddings")
            
            # Initialize ChromaDB client
            self.client = self._init_chromadb()
            self.collection = self._get_or_create_collection()
            
            # Initialize embedding model
            self.embedding_model = self._load_embedding_model()
            
        except Exception as e:
            self.dependencies_available = False
            raise ImportError(f"Failed to initialize vector database: {e}")
    
    def _init_fallback_mode(self):
        """Initialize fallback mode with simple JSON storage."""
        self.storage_path = Path(self.config.path) / "fallback_vectors.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    self.fallback_data = json.load(f)
            except:
                self.fallback_data = {"documents": {}, "embeddings": {}}
        else:
            self.fallback_data = {"documents": {}, "embeddings": {}}
        
        self.logger.info("Vector database running in fallback mode (JSON storage)")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client."""
        if not self.dependencies_available:
            return None
            
        try:
            # Try the ephemeral client for now - works with most versions
            client = self.chromadb.EphemeralClient()
            self.logger.info("ChromaDB EphemeralClient initialized (data will not persist)")
            return client
        except Exception as e:
            self.logger.error(f"ChromaDB initialization failed: {e}")
            # Fall back to JSON storage
            self.dependencies_available = False
            self._init_fallback_mode()
            return None
    
    def _get_or_create_collection(self):
        """Get or create the document collection."""
        if not self.dependencies_available or not self.client:
            return None
            
        try:
            collection = self.client.get_collection(name=self.config.collection_name)
            self.logger.info(f"Using existing collection: {self.config.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"description": "PDF document embeddings for semantic analysis"}
            )
            self.logger.info(f"Created new collection: {self.config.collection_name}")
        
        return collection
    
    def _load_embedding_model(self):
        """Load the embedding model."""
        if not self.dependencies_available:
            return None
        
        if hasattr(self, 'use_real_embeddings') and self.use_real_embeddings:
            try:
                model = self.SentenceTransformer(self.config.embedding_model)
                self.logger.info(f"Loaded embedding model: {self.config.embedding_model}")
                return model
            except Exception as e:
                self.logger.error(f"Failed to load real embedding model: {e}")
                self.use_real_embeddings = False
        
        # Use hash-based fallback
        class HashEmbeddingModel:
            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                embeddings = []
                for text in texts:
                    # Create a simple 384-dimensional embedding based on text hash
                    hash_obj = hashlib.md5(text.encode())
                    hash_int = int(hash_obj.hexdigest(), 16)
                    embedding = [(hash_int >> i) & 1 for i in range(384)]
                    embedding = [float(x) for x in embedding]  # Convert to float
                    embeddings.append(embedding)
                import numpy as np
                return np.array(embeddings)
        
        self.logger.info("Using hash-based embedding fallback")
        return HashEmbeddingModel()
    
    def add_document(self, analysis_result: AnalysisResult, full_text: str) -> bool:
        """Add document embeddings."""
        try:
            document_id = self._create_document_id(analysis_result)
            
            if not self.dependencies_available:
                # Fallback mode - store in JSON
                self.fallback_data["documents"][document_id] = {
                    "filename": analysis_result.filename,
                    "full_text": full_text[:2000],  # Store first 2000 chars
                    "topics": [topic.topic for topic in analysis_result.topics],
                    "metadata": {
                        "author": analysis_result.author,
                        "title": analysis_result.title,
                        "page_count": analysis_result.page_count,
                        "word_count": analysis_result.word_count,
                        "timestamp": analysis_result.timestamp.isoformat()
                    }
                }
                self._save_fallback_data()
                self.logger.info(f"Added document to fallback storage: {analysis_result.filename}")
                return True
            
            # Regular ChromaDB storage
            text_chunks = self._chunk_text_for_embeddings(full_text)
            embeddings = self.embedding_model.encode(text_chunks)
            
            # Prepare metadata and IDs
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(text_chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                metadata = {
                    "document_id": document_id,
                    "filename": analysis_result.filename,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "topics": ", ".join([topic.topic for topic in analysis_result.topics]),
                    "author": analysis_result.author or "Unknown",
                    "timestamp": analysis_result.timestamp.isoformat(),
                }
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=text_chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added document to vector DB: {analysis_result.filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            return False
    
    def find_similar_documents(self, query_text: str, limit: int = 5, 
                             threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Find documents similar to query text."""
        try:
            if not self.dependencies_available:
                # Fallback mode - simple text matching
                return self._fallback_text_search(query_text, limit)
            
            # Real semantic search
            query_embedding = self.embedding_model.encode([query_text])
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=limit * 2,
                include=['metadatas', 'distances', 'documents']
            )
            
            similar_docs = []
            seen_documents = set()
            
            for distance, metadata, document in zip(
                results['distances'][0], 
                results['metadatas'][0], 
                results['documents'][0]
            ):
                similarity_score = 1 - distance
                doc_id = metadata['document_id']
                
                if doc_id in seen_documents:
                    continue
                seen_documents.add(doc_id)
                
                similar_docs.append({
                    'filename': metadata['filename'],
                    'similarity_score': round(similarity_score, 4),
                    'topics': metadata.get('topics', ''),
                    'author': metadata.get('author', 'Unknown'),
                    'matched_text': document[:200] + "..." if len(document) > 200 else document
                })
                
                if len(similar_docs) >= limit:
                    break
            
            return similar_docs
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    def find_similar_to_document(self, filename: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a specific document."""
        if not self.dependencies_available:
            return self._fallback_similar_docs(filename, limit)
        
        # Implementation for real similarity search
        return []
    
    def cluster_documents(self, num_clusters: int = 5) -> Dict[str, Any]:
        """Perform document clustering."""
        if not self.dependencies_available:
            return self._fallback_clustering(num_clusters)
        
        # Real clustering implementation would go here
        return {"error": "Clustering not implemented in this version"}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.dependencies_available:
            doc_count = len(self.fallback_data.get("documents", {}))
            return {
                'total_documents': doc_count,
                'mode': 'fallback',
                'database_path': str(self.storage_path)
            }
        
        try:
            if self.collection:
                count = self.collection.count()
                return {
                    'total_chunks': count,
                    'mode': 'chromadb',
                    'collection_name': self.config.collection_name
                }
        except:
            pass
        
        return {'error': 'Unable to get stats'}
    
    def _fallback_text_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Simple text-based search for fallback mode."""
        results = []
        query_lower = query.lower()
        
        for doc_id, doc_data in self.fallback_data.get("documents", {}).items():
            # Simple text matching
            text = doc_data.get("full_text", "").lower()
            topics_text = " ".join(doc_data.get("topics", [])).lower()
            
            # Calculate simple relevance score
            score = 0.0
            query_words = query_lower.split()
            
            for word in query_words:
                if word in text:
                    score += 0.3
                if word in topics_text:
                    score += 0.7
            
            if score > 0:
                results.append({
                    'filename': doc_data['filename'],
                    'similarity_score': min(score, 1.0),
                    'topics': ", ".join(doc_data.get('topics', [])),
                    'author': doc_data['metadata'].get('author', 'Unknown'),
                    'matched_text': doc_data['full_text'][:200] + "..."
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:limit]
    
    def _fallback_similar_docs(self, filename: str, limit: int) -> List[Dict[str, Any]]:
        """Find similar documents in fallback mode."""
        # Simple implementation - find docs with similar topics
        target_doc = None
        for doc_data in self.fallback_data.get("documents", {}).values():
            if doc_data['filename'] == filename:
                target_doc = doc_data
                break
        
        if not target_doc:
            return []
        
        target_topics = set(target_doc.get('topics', []))
        results = []
        
        for doc_data in self.fallback_data.get("documents", {}).values():
            if doc_data['filename'] == filename:
                continue
            
            doc_topics = set(doc_data.get('topics', []))
            overlap = len(target_topics & doc_topics)
            total = len(target_topics | doc_topics)
            
            if total > 0:
                similarity = overlap / total
                if similarity > 0:
                    results.append({
                        'filename': doc_data['filename'],
                        'similarity_score': similarity,
                        'topics': ", ".join(doc_data.get('topics', [])),
                        'author': doc_data['metadata'].get('author', 'Unknown')
                    })
        
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:limit]
    
    def _fallback_clustering(self, num_clusters: int) -> Dict[str, Any]:
        """Simple clustering in fallback mode."""
        docs = list(self.fallback_data.get("documents", {}).values())
        
        if len(docs) == 0:
            return {"error": "No documents found"}
        
        # Simple topic-based clustering
        topic_groups = {}
        for doc in docs:
            topics = doc.get('topics', [])
            if topics:
                main_topic = topics[0]  # Use first topic as cluster key
                if main_topic not in topic_groups:
                    topic_groups[main_topic] = []
                topic_groups[main_topic].append(doc)
        
        clusters = {}
        for i, (topic, docs) in enumerate(topic_groups.items()):
            if i >= num_clusters:
                break
            clusters[i] = {
                'size': len(docs),
                'documents': [{'filename': d['filename'], 'topics': ", ".join(d.get('topics', []))} for d in docs],
                'representative_topic': topic
            }
        
        return {
            'total_documents': len(docs),
            'num_clusters': len(clusters),
            'clusters': clusters,
            'mode': 'fallback'
        }
    
    def _chunk_text_for_embeddings(self, text: str) -> List[str]:
        """Split text into chunks for embedding."""
        words = text.split()
        chunks = []
        chunk_size = self.config.max_chunk_size
        overlap = self.config.chunk_overlap
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) > 50:
                chunks.append(' '.join(chunk_words))
        
        return chunks if chunks else [text[:chunk_size * 5]]
    
    def _create_document_id(self, analysis_result: AnalysisResult) -> str:
        """Create unique document ID."""
        return analysis_result.file_hash[:16]
    
    def _save_fallback_data(self):
        """Save fallback data to JSON file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.fallback_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save fallback data: {e}")
    
    def reset_collection(self):
        """Reset the collection."""
        if not self.dependencies_available:
            self.fallback_data = {"documents": {}, "embeddings": {}}
            self._save_fallback_data()
            self.logger.info("Fallback storage reset")
        else:
            # Reset ChromaDB collection
            try:
                self.client.delete_collection(name=self.config.collection_name)
                self.collection = self._get_or_create_collection()
                self.logger.info("ChromaDB collection reset")
            except Exception as e:
                self.logger.error(f"Failed to reset collection: {e}")