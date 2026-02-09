
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class AuditService:
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "rag_audit_log.jsonl"
        
        # Persistence: Do NOT clear log file on startup anymore
        # if self.log_file.exists():
        #     try:
        #         # Truncate file
        #         with open(self.log_file, "w", encoding="utf-8") as f:
        #             pass
        #         logger.info("Cleared audit log for new session")
        #     except Exception as e:
        #         logger.error(f"Failed to clear audit log: {e}")

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log an event to the audit log file.
        """
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "data": data
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

    def log_rag_trace(
        self,
        query: str,
        initial_retrieval_count: int,
        filtered_count: int,
        reranked_chunks: List[Dict],
        selected_chunks: List[Dict],
        context_used: str,
        llm_response: str,
        generation_time_ms: float,
        models_info: Dict[str, str],
        file_id: Optional[str] = None,
        folder_id: Optional[str] = None
    ):
        """
        Specialized logger for RAG traces
        """
        data = {
            "query": query,
            "context_filter": {
                "file_id": file_id,
                "folder_id": folder_id
            },
            "metrics": {
                "initial_retrieval_count": initial_retrieval_count,
                "filtered_count": filtered_count,
                "generation_time_ms": generation_time_ms
            },
            "reranking": [
                {
                    "text_snippet": c.get("text", "")[:100] + "...",
                    "file_id": c.get("file_id"),
                    "rerank_score": c.get("rerank_score")
                }
                for c in reranked_chunks
            ],
            "selected_context_chunks": [
                {
                    "text_snippet": c.get("text", "")[:100] + "...",
                    "file_id": c.get("file_id"),
                    "page": c.get("page")
                }
                for c in selected_chunks
            ],
            "full_context_length": len(context_used),
            "llm_response": llm_response,
            "models": models_info
        }
        self.log_event("RAG_TRACE", data)

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent trace events from the audit log.
        Supports RAG_TRACE, CHAT_TRACE, IMAGE_TRACE, AUDIO_TRACE.
        """
        history = []
        if not self.log_file.exists():
            return history

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Process in reverse order to get newest first
            for line in reversed(lines):
                if len(history) >= limit:
                    break
                
                try:
                    entry = json.loads(line)
                    event_type = entry.get("event_type")
                    data = entry.get("data", {})
                    
                    if event_type == "RAG_TRACE":
                        history.append({
                            "timestamp": entry.get("timestamp"),
                            "query": data.get("query"),
                            "answer": data.get("llm_response"),
                            "sources": data.get("selected_context_chunks", [])
                        })
                    elif event_type == "CHAT_TRACE":
                        history.append({
                            "timestamp": entry.get("timestamp"),
                            "query": data.get("message"),
                            "answer": data.get("response"),
                            "sources": [] # No sources for generic chat
                        })
                    elif event_type == "IMAGE_TRACE":
                        history.append({
                            "timestamp": entry.get("timestamp"),
                            "query": data.get("question"),
                            "answer": data.get("answer"),
                            "sources": data.get("sources", [])
                        })
                    elif event_type == "AUDIO_TRACE":
                        history.append({
                            "timestamp": entry.get("timestamp"),
                            "query": data.get("question"),
                            "answer": data.get("answer"),
                            "sources": data.get("sources", [])
                        })
                        
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to read history: {e}")
            
        return history

audit_service = AuditService()
