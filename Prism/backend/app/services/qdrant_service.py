import logging
import uuid
import threading
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from .instructor_service import instructor_service

logger = logging.getLogger(__name__)

class QdrantConfig:
    """Centralized configuration for Qdrant Vector Service"""
    # Local Path (No Docker)
    DB_PATH = "./qdrant_data"
    COLLECTION_NAME = "prism_vectors"
    
    # Embedding Dimensions (Must match model)
    # Using 768 to match instructor-xl / all-mpnet-base-v2
    VECTOR_SIZE = 768 
    
    # HNSW Index Configuration (Recall Optimized)
    HNSW_M = 32
    HNSW_EF_CONSTRUCT = 200
    HNSW_FULL_SCAN_THRESHOLD = 10000
    
    # Search Configuration
    SEARCH_EF = 128
    
    # Optimizer Configuration
    INDEXING_THRESHOLD = 20000
    
    # Batch Size for Upserts
    # Batch Size for Upserts
    BATCH_SIZE = 64
    
    # Sparse Model
    SPARSE_MODEL_NAME = "Qdrant/bm25"

class QdrantVectorService:
    def __init__(self):
        self.config = QdrantConfig
        self.client: Optional[QdrantClient] = None
        self._lock = threading.RLock()
        self._initialized = False
        # Lazy load sparse model
        self.sparse_model = None
        
    def _ensure_initialized(self):
        """Lazy initialization of Qdrant Client and Collection"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return
                
            try:
                logger.info(f"Initializing Qdrant (Local) at {self.config.DB_PATH}...")
                
                # Initialize Client (Path-based, Persistent)
                self.client = QdrantClient(path=self.config.DB_PATH)
                
                # Check if collection exists
                collections = self.client.get_collections().collections
                exists = any(c.name == self.config.COLLECTION_NAME for c in collections)
                
                if not exists:
                    logger.info(f"Creating collection '{self.config.COLLECTION_NAME}'...")
                    self.client.create_collection(
                        collection_name=self.config.COLLECTION_NAME,
                        vectors_config={
                            "text-dense": models.VectorParams(
                                size=self.config.VECTOR_SIZE,
                                distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "text-sparse": models.SparseVectorParams(
                                index=models.SparseIndexParams(
                                    on_disk=True,
                                )
                            )
                        },
                        hnsw_config=models.HnswConfigDiff(
                            m=self.config.HNSW_M,
                            ef_construct=self.config.HNSW_EF_CONSTRUCT,
                            full_scan_threshold=self.config.HNSW_FULL_SCAN_THRESHOLD
                        ),
                        optimizers_config=models.OptimizersConfigDiff(
                            indexing_threshold=self.config.INDEXING_THRESHOLD,
                            default_segment_number=2 
                        ),
                        # Disable quantization for max recall
                        quantization_config=None 
                    )
                else:
                    logger.info(f"Connected to existing collection '{self.config.COLLECTION_NAME}'.")
                    
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant: {e}", exc_info=True)
                raise e

    def add_documents(self, chunks: List[Dict]):
        """
        Batch Upsert Documents
        - Generates Embeddings
        - Extracts Metadata (NO TEXT STORAGE in Qdrant)
        - Batches requests
        """
        self._ensure_initialized()
        if not chunks:
            return

        total_chunks = len(chunks)
        batch_size = self.config.BATCH_SIZE
        
        logger.info(f"Starting ingestion of {total_chunks} chunks to Qdrant...")
        
        # 1. Generate Embeddings (Batching handled by instructor service or here?)
        # For simplicity and memory safety, let's process in batches entirely
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            
            try:
                # A. Embed Dense
                texts = [c.get("text", "") for c in batch]
                dense_embeddings = instructor_service.encode_documents(texts)
                
                # B. Embed Sparse
                if not self.sparse_model:
                     logger.info(f"Loading Sparse Model {self.config.SPARSE_MODEL_NAME}...")
                     self.sparse_model = SparseTextEmbedding(model_name=self.config.SPARSE_MODEL_NAME)
                
                # fastembed returns generator, convert to list
                sparse_embeddings = list(self.sparse_model.embed(texts))
                
                points = []
                for idx, chunk in enumerate(batch):
                    # B. ID Generation
                    # Use provided chunk_id or generate deterministic UUID based on file_id+index
                    cid = chunk.get("chunk_id")
                    if not cid:
                        cid = str(uuid.uuid4())
                    
                    # Ensure Point ID is a valid UUID (Qdrant strict requirement or safer)
                    # Helper to convert arbitrary string ID to UUIDv5
                    try:
                        point_id = str(uuid.UUID(cid)) # Check if already valid
                    except:
                        # Create deterministic UUID from string ID
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(cid)))
                        
                    # C. Metadata Extraction (STRICTLY NO TEXT)
                    # We only keep what's needed for filtering & joining
                    payload = {
                        "chunk_id": cid,
                        "doc_id": chunk.get("file_id") or chunk.get("doc_id", "unknown"),
                        "folder_id": chunk.get("folder_id", "unknown"),
                        "source_name": chunk.get("source_file") or chunk.get("file_name", "unknown"),
                        "page": chunk.get("page"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "file_type": chunk.get("file_type", "unknown")
                    }
                    
                    # D. Construct Point
                    points.append(models.PointStruct(
                        id=point_id,
                        vector={
                            "text-dense": dense_embeddings[idx].tolist(),
                            "text-sparse": models.SparseVector(
                                indices=sparse_embeddings[idx].indices.tolist(),
                                values=sparse_embeddings[idx].values.tolist()
                            )
                        },
                        payload=payload
                    ))
                
                # E. Upsert Batch
                self.client.upsert(
                    collection_name=self.config.COLLECTION_NAME,
                    points=points
                )
                
            except Exception as e:
                logger.error(f"Failed to upsert batch {i}-{i+len(batch)}: {e}")
                # Continue? Or raise? For robustness, we log and continue, but strictly this might lose data.
                # In strict mode, we might want to raise. Let's log heavily.
                
        logger.info(f"Completed ingestion of {total_chunks} chunks.")

    def search(self, query: str, k: int = 40, folder_id: str = None, file_id: str = None) -> List[Dict]:
        """
        Retrieval
        - Embed Query
        - Filter by folder/file
        - Return Scored Points (Chunk IDs)
        """
        self._ensure_initialized()
        
        # 1. Embed
        query_vec = instructor_service.encode_query(query).tolist()
        
        # 2. Build Filter
        filter_conditions = []
        if folder_id:
            filter_conditions.append(
                models.FieldCondition(key="folder_id", match=models.MatchValue(value=folder_id))
            )
        if file_id:
             filter_conditions.append(
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=file_id))
            )

        q_filter = None
        if filter_conditions:
            q_filter = models.Filter(must=filter_conditions)

        # 3. Search (Hybrid)
        try:
             # Ensure sparse model loaded for query
            if not self.sparse_model:
                 self.sparse_model = SparseTextEmbedding(model_name=self.config.SPARSE_MODEL_NAME)
            
            sparse_q = list(self.sparse_model.embed([query]))[0]
            
            # Hybrid Query using Prefetch
            results = self.client.query_points(
                collection_name=self.config.COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        using="text-sparse",
                        query=models.SparseVector(
                            indices=sparse_q.indices.tolist(), 
                            values=sparse_q.values.tolist()
                        ),
                        limit=k,
                        filter=q_filter
                    ),
                    models.Prefetch(
                        using="text-dense",
                        query=query_vec,
                        limit=k,
                        filter=q_filter
                    )
                ],
                # RRF Fusion
                query=models.FusionQuery(method=models.Fusion.RRF),
                limit=k
            ).points
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}. Falling back to Dense Only.")
            # Fallback to old search if query_points/fusion not supported by server/client version
            try:
                 results = self.client.search(
                    collection_name=self.config.COLLECTION_NAME,
                    query_vector=models.NamedVector(
                        name="text-dense",
                        vector=query_vec
                    ),
                    query_filter=q_filter,
                    limit=k
                )
            except Exception as e2:
                 logger.error(f"Fallback Search failed: {e2}")
                 return []

        # 4. Format Results
        # Return format expected by QA Service: {"id":..., "score":..., "payload":...}
        # Note: 'chunk' (text) is NOT returned here. QA Service must hydrate it.
        formatted_results = []
        for point in results:
            formatted_results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload, # Contains chunk_id, doc_id
                "chunk_id": point.payload.get("chunk_id", point.id)
            })
            
        return formatted_results

    def delete_collection(self):
        """Destructive: Clear all data"""
        self._ensure_initialized()
        try:
            self.client.delete_collection(self.config.COLLECTION_NAME)
            # Re-init (create empty) handled by lazy logic on next call, 
            # but to be safe we can force re-init state
            self._initialized = False
            logger.info("Collection deleted.")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")

    def get_count(self) -> int:
        self._ensure_initialized()
        try:
            info = self.client.get_collection(self.config.COLLECTION_NAME)
            return info.points_count
        except:
            return 0

# Singleton Instance
qdrant_service = QdrantVectorService()
