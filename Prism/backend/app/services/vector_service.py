import os
import logging
import threading
import uuid
import shutil
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from .instructor_service import instructor_service

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, data_dir: str = "data/chroma_db"):
        self.data_dir = Path(data_dir)
        # self.data_dir.mkdir(parents=True, exist_ok=True) # Chroma handles this
        
        self.collection_name = "prism_vectors"
        
        # Lazy initialization
        self.client = None
        self.collection = None
        self._lock = threading.RLock()
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return
                
            try:
                import chromadb
                from chromadb.config import Settings
                
                logger.info(f"Initializing ChromaDB at {self.data_dir}...")
                
                self.client = chromadb.PersistentClient(
                    path=str(self.data_dir),
                    settings=Settings(anonymized_telemetry=False)
                )
                
                # Check/Create Collection
                # We use cosine distance. 
                # Note: Chroma defaults to L2. We must specify metadata={"hnsw:space": "cosine"}
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Connected to Chroma collection '{self.collection_name}'.")

                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                raise e

    def add_documents(self, chunks: List[Dict]):
        """
        Ingestion Logic:
        1. Embed with Instructor (all-mpnet-base-v2)
        2. Create IDs and Metadata
        3. Upsert to Chroma
        """
        self._ensure_initialized()
        if not chunks:
            return
            
        # 1. Embed
        texts = [c.get("text", "") for c in chunks]
        embeddings = instructor_service.encode_documents(texts)
        
        ids = []
        metadatas = []
        documents = []
        
        with self._lock:
            for i, chunk in enumerate(chunks):
                # Ensure clean ID
                cid = chunk.get("chunk_id")
                if not cid:
                    cid = str(uuid.uuid4())
                # Verify string ID
                point_id = str(cid)
                
                # Prepare Metadata
                # Chroma requires flat dicts (str/int/float/bool) typically.
                # We need to extract metadata from payload and flatten it.
                meta_raw = chunk.copy()
                if "metadata" in meta_raw:
                    nested = meta_raw.pop("metadata")
                    if isinstance(nested, dict):
                        meta_raw.update(nested)
                
                # Sanitize metadata for Chroma (no None values, lists, etc if not supported)
                clean_meta = {}
                for k, v in meta_raw.items():
                    if v is None:
                        continue 
                    if isinstance(v, (str, int, float, bool)):
                         clean_meta[k] = v
                    else:
                        # Fallback for complex types -> stringify
                        clean_meta[k] = str(v)
                
                # Ensure core fields
                if "doc_id" not in clean_meta: clean_meta["doc_id"] = chunk.get("file_id", "unknown")
                if "folder_id" not in clean_meta: clean_meta["folder_id"] = chunk.get("folder_id", "unknown")

                # Note: We do NOT authorize storing the full text in metadata if it's huge, 
                # but user requirement says "Persistent local storage". 
                # Storing text in 'documents' list is standard Chroma.
                
                ids.append(point_id)
                embeddings_list = embeddings[i].tolist()
                metadatas.append(clean_meta)
                documents.append(texts[i])

            # Upsert Batch
            try:
                # Upsert handles update-or-insert
                self.collection.upsert(
                    ids=ids,
                    embeddings=[e.tolist() for e in embeddings], # numpy -> list[float]
                    metadatas=metadatas,
                    documents=documents
                )
                logger.info(f"Successfully ingested {len(ids)} chunks into ChromaDB.")
            except Exception as e:
                logger.error(f"Failed to upsert to ChromaDB: {e}")
                raise e

    def search(self, query: str, k: int = 500, folder_id: str = None, file_id: str = None) -> List[Dict]:
        """
        Retrieval Logic:
        1. Embed query
        2. Build Filters (Folder/File scope)
        3. Vector Search
        """
        self._ensure_initialized()

        # 1. Embed
        query_vec = instructor_service.encode_query(query).tolist()
        
        # 2. Build Filters
        # Chroma format: {"metadata_field": "value"} or {"$and": [...]}
        where_clause = {}
        conditions = []

        if folder_id:
             conditions.append({"folder_id": folder_id})
        
        if file_id:
             conditions.append({"doc_id": file_id})

        if len(conditions) > 1:
            where_clause = {"$and": conditions}
        elif len(conditions) == 1:
            where_clause = conditions[0]
        else:
            where_clause = None # No filter
            
        # 3. Search
        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=k,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
        except Exception as e:
            logger.error(f"Chroma search failed: {e}")
            return []
            
        # 4. Map to Standard Format
        # results is a dict of lists (batch format)
        # results['ids'][0] -> list of ids for first query
        if not results['ids']:
            return []
            
        final_results = []
        ids = results['ids'][0]
        metas = results['metadatas'][0]
        docs = results['documents'][0]
        dists = results['distances'][0]
        
        for i in range(len(ids)):
            # Reconstruct 'chunk'
            chunk = metas[i].copy()
            chunk['text'] = docs[i]
            chunk['chunk_id'] = ids[i]
            
            # Chroma returns DISTANCE. We usually want SIMILARITY or SCORE.
            # But the rest of the app might handle raw numbers or need consistency.
            # Cosine distance: 0=identical, 2=opposite.
            # Score (conventionally) = 1 - distance, or just passing distance.
            # Let's pass the raw distance as 'score' but log it, or invert it?
            # User requirement: "Apply metadata filtering... Retrieve high recall... Apply cross-encoder".
            # Cross-encoder doesn't care about the initial score, just the candidates.
            
            final_results.append({
                "chunk": chunk,
                "score": 1.0 - dists[i], # Explicitly returning similarity approximation
                "id": ids[i]
            })
            
        return final_results

    def clear(self):
        self._ensure_initialized()
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'.")
            
            # Recreate
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB: {e}")

vector_service = VectorStoreService()
