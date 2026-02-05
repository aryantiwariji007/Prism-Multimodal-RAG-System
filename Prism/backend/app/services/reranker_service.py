
# from sentence_transformers import CrossEncoder
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RerankerService:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        # Lazy load model on first use
        # self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Reranker Model: {self.model_name} on {device}...")
            self.model = CrossEncoder(self.model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to load Reranker Model: {e}. Reranking will be disabled.")
            self.model = None
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5, threshold: float = -10.0) -> List[Dict]:
        """
        Rerank a list of candidates based on query relevance.
        candidates: List of dicts, each must have 'chunk' -> 'text'.
        threshold: Score threshold. MS-MARCO logits < -2.0 usually indicate irrelevance.
        Returns: Sorted list of candidates that pass the threshold.
        """
        if not candidates:
            return []
            
        # Prepare pairs for Cross Encoder
        pairs = []
        valid_candidates = []
        
        for cand in candidates:
            chunk_text = cand.get("chunk", {}).get("text", "")
            if chunk_text:
                pairs.append([query, chunk_text])
                valid_candidates.append(cand)
                
        if not pairs:
            return []
            
        if not self.model:
            self._load_model()
            
        if not self.model:
            return candidates[:top_k]
            
        scores = self.model.predict(pairs)
        
        # Attach scores and filter
        scored_results = []
        for i, cand in enumerate(valid_candidates):
            score = float(scores[i])
            cand["rerank_score"] = score
            if score >= threshold:
                scored_results.append(cand)
            
        # Sort by rerank_score descending
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        if scored_results:
            logger.info(f"Reranking: kept {len(scored_results)}/{len(valid_candidates)}. Top score: {scored_results[0]['rerank_score']:.2f}")
        else:
             logger.info(f"Reranking: All {len(valid_candidates)} candidates below threshold {threshold}.")

        return scored_results[:top_k]

# Singleton
reranker_service = RerankerService()
