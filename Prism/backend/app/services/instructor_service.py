import logging
import numpy as np
import threading
from typing import List, Union

# Lazy import for torch and SentenceTransformer
# import torch
# from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class InstructorEmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = None):
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.model = None
        self.dimension = 768
        self.is_instructor = "instructor" in model_name.lower()
        self._lock = threading.Lock()
        # Lazy load on first use
        # self._load_model()
            
    def _load_model(self):
        if self.model:
            return
            
        with self._lock:
            if self.model:
                return

            logger.info(f"Initializing Embedding Service with {self.model_name} on {self.device}...")
            try:
                # We use SentenceTransformer for all-mpnet-base-v2
                import torch
                from sentence_transformers import SentenceTransformer
                
                self.model = SentenceTransformer(self.model_name)
                self.model.to(self.device)
                logger.info(f"Model {self.model_name} loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise e

    def encode_documents(self, texts: List[str], instruction: str = "Represent the document for retrieval:") -> np.ndarray:
        """
        Embed documents. Supports Instructor-style instructions if using an instructor model.
        Returns a normalized (L2) numpy array.
        """
        if self.model is None:
            self._load_model()

        if self.is_instructor:
            data = [[instruction, text] for text in texts]
        else:
            data = texts
        
        logger.debug(f"Encoding {len(texts)} documents with {self.model.get_max_seq_length()} max tokens")
        embeddings = self.model.encode(data)
        
        # Explicit L2 Normalization (already usually handled by ST but we ensure it here)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-9)
        
        return normalized_embeddings.astype('float32')

    def encode_query(self, query: str, instruction: str = "Represent the question for retrieval:") -> np.ndarray:
        """
        Embed a single query.
        Returns a normalized (L2) numpy array.
        """
        if self.model is None:
            self._load_model()

        if self.is_instructor:
            data = [[instruction, query]]
        else:
            data = query
        
        logger.debug(f"Encoding query")
        embedding = self.model.encode(data)
        
        # Explicit L2 Normalization
        if len(embedding.shape) == 1:
            norm = np.linalg.norm(embedding)
            normalized_embedding = embedding / (norm + 1e-9)
            return normalized_embedding.astype('float32')
        else:
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            normalized_embedding = embedding / (norm + 1e-9)
            return normalized_embedding[0].astype('float32')



# Singleton instance
instructor_service = InstructorEmbeddingService()
