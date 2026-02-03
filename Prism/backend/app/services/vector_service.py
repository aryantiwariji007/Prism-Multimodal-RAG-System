import os
import faiss
import numpy as np
import pickle
import logging
import ollama
import threading
from typing import List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self, data_dir: str = "data/vector_store", model_name="nomic-embed-text"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.data_dir / "faiss_index.bin"
        self.metadata_path = self.data_dir / "metadata.pkl"
        
        self.model_name = model_name
        # nomic-embed-text is 768d, mxbai-embed-large is 1024d, llama3.2 is ?
        # We assume nomic-embed-text (768)
        self.dimension = 768 
        if model_name != "nomic-embed-text":
             # Fallback estimation or force
             pass

        self.index = None
        self.metadata: List[Dict] = []
        self._lock = threading.RLock()
        
        # Pull model if needed (blocking, but safe)
        try:
             logger.info(f"Ensuring embedding model {self.model_name} is available...")
             ollama.pull(self.model_name)
        except Exception as e:
             logger.error(f"Failed to pull model {self.model_name}: {e}")
        
        self._load_store()

    def _load_store(self):
        with self._lock:
            if self.index_path.exists() and self.metadata_path.exists():
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    with open(self.metadata_path, "rb") as f:
                        self.metadata = pickle.load(f)
                    
                    # Check dimension compatibility
                    if self.index.d != self.dimension:
                        logger.warning(f"Index dimension mismatch ({self.index.d} vs {self.dimension}). Resetting index.")
                        self._create_new_index()
                    else:
                        logger.info(f"Loaded Vector Store: {self.index.ntotal} vectors.")
                except Exception as e:
                    logger.error(f"Failed to load vector store: {e}. Creating new.")
                    self._create_new_index()
            else:
                self._create_new_index()

    def _create_new_index(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        logger.info(f"Created new FAISS index (dim={self.dimension}).")

    def save_store(self):
        with self._lock:
            try:
                faiss.write_index(self.index, str(self.index_path))
                with open(self.metadata_path, "wb") as f:
                    pickle.dump(self.metadata, f)
                logger.info("Vector Store saved.")
            except Exception as e:
                logger.error(f"Error saving vector store: {e}")

    def _get_embedding(self, text: str, prefix: str = "") -> np.ndarray:
        try:
            # fast embed models often need a prefix
            # nomic-embed-text: "search_query: " for questions, "search_document: " for docs
            input_text = f"{prefix}{text}"
            response = ollama.embeddings(model=self.model_name, prompt=input_text)
            embedding = response.get("embedding")
            if not embedding:
                 raise ValueError("No embedding returned")
            return np.array(embedding, dtype="float32")
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros(self.dimension, dtype="float32")

    def add_documents(self, chunks: List[Dict]):
        if not chunks:
            return
            
        embeddings_list = []
        valid_chunks = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue
            # PREFIX ADDED HERE for documents
            emb = self._get_embedding(text, prefix="search_document: ")
            embeddings_list.append(emb)
            valid_chunks.append(chunk)

        if not embeddings_list:
            return

        with self._lock:
            embeddings = np.array(embeddings_list)
            self.index.add(embeddings)
            
            self.metadata.extend(valid_chunks)
            self.save_store()
            logger.info(f"Added {len(valid_chunks)} documents to Vector Store using {self.model_name}.")

    def search(self, query: str, k: int = 50) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
            
        with self._lock:
            # PREFIX ADDED HERE for query
            query_embedding = self._get_embedding(query, prefix="search_query: ").reshape(1, -1)
            # Fetch more candidates to account for soft-deleted ones
            fetch_k = min(k * 2 + 100, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, fetch_k)
            
            results = []
            if indices.size > 0:
                 for i, idx in enumerate(indices[0]):
                    if idx == -1 or idx >= len(self.metadata):
                        continue
                        
                    item = self.metadata[idx]
                    # Soft delete check
                    if item.get("deleted", False):
                        continue
                        
                    results.append({
                        "chunk": item,
                        "score": float(distances[0][i])
                    })
                    if len(results) >= k:
                        break
                
            return results

    def delete_document(self, file_id: str):
        """Soft delete chunks belonging to a file_id"""
        with self._lock:
            count = 0
            for item in self.metadata:
                if item.get("file_id") == file_id:
                    item["deleted"] = True
                    count += 1
            if count > 0:
                self.save_store()
                logger.info(f"Soft deleted {count} chunks for file_id {file_id}")

    def compact(self):
        """Rebuild index to remove soft-deleted items (Maintenance)"""
        with self._lock:
            logger.info("Starting vector store compaction...")
            valid_items = [item for item in self.metadata if not item.get("deleted", False)]
            if len(valid_items) == len(self.metadata):
                logger.info("No deleted items found. Compaction skipped.")
                return

            new_index = faiss.IndexFlatL2(self.dimension)
            
            # Re-add all vectors
            # This requires us to HAVE the vectors. 
            # FAISS IndexFlatL2 stores them. methods like reconstruct_n exist.
            # But getting them out of flat index one by one might be slow or we can use index.reconstruct_n(0, ntotal)
            
            try:
                if self.index.ntotal > 0:
                     # Get all vectors
                     all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                     
                     # Filter
                     valid_indices = [i for i, item in enumerate(self.metadata) if not item.get("deleted", False)]
                     
                     if valid_indices:
                         valid_vectors = all_vectors[valid_indices]
                         new_index.add(valid_vectors)
                     
                self.index = new_index
                self.metadata = valid_items
                self.save_store()
                logger.info(f"Compaction complete. New size: {self.index.ntotal}")
                
            except Exception as e:
                logger.error(f"Compaction failed: {e}")

    def clear(self):
        with self._lock:
            self._create_new_index()
            self.save_store()

vector_service = VectorStoreService()
