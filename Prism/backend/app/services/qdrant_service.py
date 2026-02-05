import logging
import uuid
import threading
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from .instructor_service import instructor_service

logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================

class QdrantConfig:
    DB_PATH = "./qdrant_data"
    COLLECTION_NAME = "prism_vectors"

    VECTOR_SIZE = 768  # all-mpnet-base-v2 / instructor-xl

    # HNSW (recall-optimized)
    HNSW_M = 32
    HNSW_EF_CONSTRUCT = 200
    HNSW_FULL_SCAN_THRESHOLD = 10000

    # Search
    SEARCH_EF = 512   # MUST be >= 2x k

    # Optimizer
    INDEXING_THRESHOLD = 100000

    # Ingestion
    BATCH_SIZE = 64

    # Sparse model
    SPARSE_MODEL_NAME = "Qdrant/bm25"


# =========================
# Vector Service
# =========================

class QdrantVectorService:
    def __init__(self):
        self.config = QdrantConfig
        self.client: Optional[QdrantClient] = None
        self._lock = threading.RLock()
        self._initialized = False
        self.sparse_model: Optional[SparseTextEmbedding] = None

    # -------------------------
    # Initialization
    # -------------------------

    def _ensure_initialized(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            logger.info(f"Initializing Qdrant at {self.config.DB_PATH}")
            self.client = QdrantClient(path=self.config.DB_PATH)

            collections = self.client.get_collections().collections
            exists = any(c.name == self.config.COLLECTION_NAME for c in collections)
            if exists:
                try:
                    # Validate schema supports sparse vectors by checking collection info
                    coll_info = self.client.get_collection(self.config.COLLECTION_NAME)
                    has_sparse = False
                    # Check if sparse_vectors config exists and contains "text-sparse"
                    if hasattr(coll_info.config.params, 'sparse_vectors') and coll_info.config.params.sparse_vectors:
                         if "text-sparse" in coll_info.config.params.sparse_vectors:
                             has_sparse = True
                    
                    if not has_sparse:
                        logger.warning(f"Collection '{self.config.COLLECTION_NAME}' missing sparse vectors. Recreating...")
                        self.client.delete_collection(self.config.COLLECTION_NAME)
                        exists = False
                except Exception as e:
                    logger.warning(f"Failed to validate collection schema: {e}. Assuming recreation needed.")
                    self.client.delete_collection(self.config.COLLECTION_NAME)
                    exists = False

            if not exists:
                logger.info(f"Creating collection '{self.config.COLLECTION_NAME}'")

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
                            index=models.SparseIndexParams(on_disk=True)
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
                    quantization_config=None
                )
            else:
                logger.info(f"Connected to existing collection '{self.config.COLLECTION_NAME}'")

            self._initialized = True

    # -------------------------
    # Ingestion
    # -------------------------

    def add_documents(self, chunks: List[Dict]):
        self._ensure_initialized()
        if not chunks:
            return

        logger.info(f"Ingesting {len(chunks)} chunks into Qdrant")

        for i in range(0, len(chunks), self.config.BATCH_SIZE):
            batch = chunks[i:i + self.config.BATCH_SIZE]

            # Deduplication: Hash chunks to avoid re-embedding identical content
            import hashlib
            
            unique_batch = []
            seen_hashes = set()
            
            for chunk in batch:
                chunk_text = chunk.get("text", "").strip()
                text_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                
                if text_hash not in seen_hashes:
                    seen_hashes.add(text_hash)
                    unique_batch.append(chunk)
            
            if not unique_batch:
                continue

            texts = [c.get("text", "") for c in unique_batch]
            dense_vectors = instructor_service.encode_documents(texts)

            with self._lock:
                if not self.sparse_model:
                    logger.info(f"Loading sparse model {self.config.SPARSE_MODEL_NAME}")
                    self.sparse_model = SparseTextEmbedding(self.config.SPARSE_MODEL_NAME)

            sparse_vectors = list(self.sparse_model.embed(texts))

            points = []
            for idx, chunk in enumerate(unique_batch):
                cid = str(chunk.get("chunk_id") or uuid.uuid4())
                try:
                    point_id = str(uuid.UUID(cid))
                except Exception:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, cid))

                payload = {
                    "chunk_id": cid,
                    "doc_id": chunk.get("file_id") or chunk.get("doc_id", "unknown"),
                    "folder_id": chunk.get("folder_id", "unknown"),
                    "source_name": chunk.get("source_file", "unknown"),
                    "page": chunk.get("page"),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "file_type": chunk.get("file_type", "unknown"),
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "text-dense": dense_vectors[idx].tolist(),
                            "text-sparse": models.SparseVector(
                                indices=sparse_vectors[idx].indices.tolist(),
                                values=sparse_vectors[idx].values.tolist()
                            )
                        },
                        payload=payload
                    )
                )

            with self._lock:
                self.client.upsert(
                    collection_name=self.config.COLLECTION_NAME,
                    points=points
                )

        logger.info("Ingestion completed")

    # -------------------------
    # Search
    # -------------------------

    def search(
        self,
        query: str,
        k: int = 40,
        folder_id: Optional[str] = None,
        file_id: Optional[str] = None
    ) -> List[Dict]:

        self._ensure_initialized()

        query_vec = instructor_service.encode_query(query).tolist()

        conditions = []
        if folder_id:
            conditions.append(models.FieldCondition(
                key="folder_id",
                match=models.MatchValue(value=folder_id)
            ))
        if file_id:
            conditions.append(models.FieldCondition(
                key="doc_id",
                match=models.MatchValue(value=file_id)
            ))

        q_filter = models.Filter(must=conditions) if conditions else None
        ef = max(self.config.SEARCH_EF, k * 2)

        # -------- Hybrid Search --------
        try:
            if not self.sparse_model:
                self.sparse_model = SparseTextEmbedding(self.config.SPARSE_MODEL_NAME)

            sparse_q = list(self.sparse_model.embed([query]))[0]

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
                        filter=q_filter,
                        params=models.SearchParams(hnsw_ef=ef)
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k
            ).points

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}. Falling back to dense-only.")

            results = self.client.query_points(
                collection_name=self.config.COLLECTION_NAME,
                query=query_vec,
                using="text-dense",
                limit=k,
                filter=q_filter,
                params=models.SearchParams(hnsw_ef=ef)
            ).points

        if not results:
            logger.error("Qdrant returned 0 results — retrieval failure")
            return []

        logger.info(f"[QDRANT] Retrieved {len(results)} points")

        return [
            {
                "id": p.id,
                "score": p.score,
                "payload": p.payload,
                "chunk_id": p.payload.get("chunk_id", p.id)
            }
            for p in results
        ]

    # -------------------------
    # Utilities
    # -------------------------

    def delete_collection(self):
        self._ensure_initialized()
        self.client.delete_collection(self.config.COLLECTION_NAME)
        self._initialized = False
        logger.warning("Qdrant collection deleted")

    def get_count(self) -> int:
        self._ensure_initialized()
        return self.client.get_collection(self.config.COLLECTION_NAME).points_count


# Singleton
qdrant_service = QdrantVectorService()
