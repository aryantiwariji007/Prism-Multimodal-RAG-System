"""
Document Q&A Service
Combines document processing with Ollama (DeepSeek) for answering questions
"""

import os
import json
import uuid
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import sys
import time

logger = logging.getLogger(__name__)

# Add the backend directory to the Python path for ingestion modules
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Imports from existing modules
# We will lazy import ingestion modules to prevent startup bottlenecks and DLL conflicts
from .llm_service import ollama_llm
from .progress_service import progress_service
from .folder_service import folder_service
from .audit_service import audit_service
from .qdrant_service import qdrant_service
from .reranker_service import reranker_service


class DocumentQAService:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize Document Q&A service
        """
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.processed_dir = self.data_dir / "processed"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # In-memory stores
        self.document_chunks: Dict[str, List[Dict]] = {}
        self.document_metadata: Dict[str, Dict] = {}
        self._chunks_cache: Dict[str, List[Dict]] = {}

        self._load_existing_documents()
        
        # Check if index is empty but we have processed docs (Migration scenario)
        # In a real migration, we'd run a script, but we can do a lazy check here if desired.
        # for now, we leave it to the external migration script.

    # ------------------------------------------------------------------
    # Document processing
    # ------------------------------------------------------------------

    def process_document_with_progress(
        self,
        file_path: str,
        file_id: str = None,
        progress_file_id: str = None
    ) -> Dict:
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            file_id = file_id or file_path.stem
            logger.info(f"Processing document: {file_path}")

            # --- LAZY IMPORTS & DLL FIX ---
            # Ensure torch is imported BEFORE any ingestion module to fix WinError 127
            try:
                import torch
            except ImportError:
                pass

            try:
                from ingestion.parse_pdf import parse_document
                from ingestion.chunker import document_chunker
            except ImportError as e:
                raise ImportError(f"Failed to import core ingestion modules: {e}")

            # Step 1: Parse document
            if progress_file_id:
                progress_service.update_progress(
                    progress_file_id, 1, "Checking document type..."
                )
            
            # Check file type
            suffix = file_path.suffix.lower()
            
            # Helper for progress
            def report_progress_step(step, msg, percent_score=0):
                if progress_file_id:
                   # Map percentage for step 1 (parsing)
                   # We reserve 90% of step 1 for the actual parsing/ingestion logic
                   progress_service.update_progress(
                       progress_file_id, step, msg, percent_score
                   )
            
            # --- Ingestion Routing ---
            chunks = []
            
            if suffix in {'.pdf', '.docx'}:
                 if progress_file_id:
                    progress_service.update_progress(
                        progress_file_id, 1, "Starting document parsing..."
                    )
                 pages = parse_document(
                    str(file_path),
                    file_id=file_id,
                    progress_file_id=progress_file_id
                 )
                 # Text needs chunking
                 if progress_file_id:
                    progress_service.update_progress(
                        progress_file_id, 2, "Starting text chunking..."
                    )
                 chunks = document_chunker.chunk_document_pages(
                    pages,
                    progress_file_id=progress_file_id
                 )

            elif suffix in {'.jpg', '.jpeg', '.png', '.webp'}:
                 # Image Ingestion
                 try:
                     from ingestion.image_ingestor import ingest_image
                     chunks = ingest_image(
                         str(file_path),
                         file_id=file_id,
                         progress_callback=report_progress_step
                     )
                 except ImportError:
                     logger.warning("Image ingestion module not available.")
                     chunks = []

            elif suffix in {'.mp3', '.wav', '.m4a'}:
                 # Audio Ingestion (Transcribe)
                 try:
                     from ingestion.audio_ingestor import ingest_audio
                     chunks = ingest_audio(
                         str(file_path),
                         file_id=file_id,
                         progress_callback=report_progress_step
                     )
                 except ImportError:
                     logger.warning("Audio ingestion module not available.")
                     chunks = []

            elif suffix in {'.mp4', '.avi', '.mov', '.mkv'}:
                 # Video Ingestion
                 try:
                    from ingestion.video_ingestor import ingest_video
                    chunks = ingest_video(
                         str(file_path),
                         file_id=file_id,
                         progress_callback=report_progress_step
                     )
                 except ImportError:
                     logger.warning("Video ingestion module not available.")
                     chunks = []

            elif suffix in {'.xlsx', '.xls', '.csv'}:
                 # Excel/CSV Ingestion
                 from ingestion.excel_ingestor import ingest_excel
                 chunks = ingest_excel(
                     str(file_path),
                     file_id=file_id,
                     progress_callback=report_progress_step
                 )

            elif suffix in {'.pptx', '.ppt'}:
                 # PowerPoint Ingestion
                 from ingestion.pptx_ingestor import ingest_pptx
                 # We can pass the wrapper, but note ingest_pptx expects a callable
                 chunks = ingest_pptx(
                     str(file_path),
                     file_id=file_id,
                     progress_callback=report_progress_step
                 )

            elif suffix in {'.txt', '.md', '.log', '.json', '.xml', '.html', '.htm'}:
                 # Text/Code Ingestion
                 if progress_file_id:
                    progress_service.update_progress(
                        progress_file_id, 1, "Reading text file..."
                    )
                 try:
                     text_content = file_path.read_text(encoding='utf-8', errors='replace')
                     chunks = document_chunker.chunk_document_pages(
                         [{"text": text_content, "page": 1, "file_id": file_id}],
                         progress_file_id=progress_file_id
                     )
                 except Exception as e:
                     logger.error(f"Failed to read text file {file_path}: {e}")
                     raise

            else:
                # Fallback for others (like PPTX) - log warning but don't crash
                logger.warning(f"File type {suffix} uploaded but ingestion not fully implemented yet.")
                chunks = [] # Empty chunks means it exists but no content indexed

            # Step 2.5: Inject Filename into Text (Fix for RAG Accuracy)
            # Prepend filename to allow retrieval by file ID/name
            for chunk in chunks:
                original_text = chunk.get("text", "")
                # Only prepend if it's not already there (safety check)
                if not original_text.startswith(f"Filename:"):
                     chunk["text"] = f"Filename: {file_path.name}\nFile ID: {file_id}\n{original_text}"

            # Step 3: Store results (Unified Store)
            if progress_file_id:
                progress_service.update_progress(
                    progress_file_id, 3, "Storing processed data..."
                )

            self.document_chunks[file_id] = chunks
            self.document_metadata[file_id] = {
                "file_id": file_id, # Added for LlamaIndex Node ID consistency
                "file_name": file_path.name,
                "file_path": str(file_path),
                "type": suffix.replace('.', ''),
                "num_chunks": len(chunks),
                "total_characters": sum(
                    len(chunk.get("text", "")) for chunk in chunks
                ),
            }
            
            # --- Vector Embeddings (Custom FAISS) ---
            if progress_file_id:
                progress_service.update_progress(
                    progress_file_id, 3, "Updating Vector Database (FAISS)...", 50
                )
            
            # Assign explicit Chunk IDs BEFORE Qdrant & Storage
            # This ensures the ID in Vector DB matches the ID in JSON storage
            for chunk in chunks:
                if not chunk.get("chunk_id"):
                    chunk["chunk_id"] = str(uuid.uuid4())

            # Add to Vector Store (Qdrant)
            qdrant_service.add_documents(chunks)

            self._save_processed_document(
                file_id,
                chunks,
                self.document_metadata[file_id]
            )

            if progress_file_id:
                progress_service.update_progress(
                    progress_file_id, 3, "Processing completed", 100
                )

            logger.info(
                f"Document processed successfully: {len(chunks)} chunks created"
            )

            # Audit Log for Ingestion
            audit_service.log_event("DOCUMENT_INGESTION", {
                "file_id": file_id,
                "file_name": file_path.name,
                "chunks_created": len(chunks),
                "chunking_strategy": "recursive_character" if suffix in {'.pdf', '.docx'} else "token/other"
            })

            return {
                "success": True,
                "file_id": file_id,
                "metadata": self.document_metadata[file_id],
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            try:
                with open("data/processing_errors.log", "a") as err_log:
                    err_log.write(f"Failed processing {file_id} ({file_path.name}): {str(e)}\n")
            except:
                pass

            if progress_file_id:
                progress_service.set_error(progress_file_id, str(e))
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def query_rewriter_agent(self, query: str) -> Dict:
        """
        Agent that decides whether to rewrite the query for better retrieval.
        """
        # Heuristic Bypass: If query is obviously specific
        # 1. Long queries
        # 2. explicit filename
        # 3. Proper names (Capitalized words that aren't the start of a sentence)
        words = query.split()
        is_name_query = any(w[0].isupper() for w in words[1:] if len(w) > 0)
        
        if len(words) > 15 or "filename:" in query.lower() or (len(words) < 5 and is_name_query):
            return {"rewrite_required": False, "rewritten_queries": []}

        prompt = f"""You are an expert Retrieval Optimization Agent.
Your responsibility is to decide WHETHER to expand or rewrite a user query before retrieval.

Trigger conditions (rewrite ONLY if one or more apply):
1. The query is very short (≤ 4–5 tokens) and ambiguous.
2. The query contains vague references (e.g., "this", "that", "it", "policy", "document").
3. The query is conversational but retrieval requires keyword-style phrasing.
4. The query lacks domain-specific terms present in corporate context.

DO NOT trigger if:
- The query already contains clear technical or domain-specific terms.
- The query explicitly references a document name, section, or identifier.

When triggered:
- Generate 2–4 semantically equivalent rewritten queries.
- Preserve the original intent exactly.
- Prefer adding synonyms, acronyms, and formal terminology.

User Query: "{query}"

Output format (JSON ONLY):
{{
  "rewrite_required": true | false,
  "reason": "<brief reason>",
  "rewritten_queries": [
    "query variant 1",
    "query variant 2"
  ]
}}
"""
        try:
            result = ollama_llm.generate_json_response(prompt, max_tokens=300)
            if not isinstance(result, dict):
                logger.warning("Query rewriter returned non-dict")
                return {"rewrite_required": False}
            return result
        except Exception as e:
            logger.error(f"Query rewriter agent failed: {e}")
            return {"rewrite_required": False}

    def answer_question(
        self,
        question: str,
        file_id: str = None,
        folder_id: str = None,
        max_chunks: int = 10
    ) -> Dict:
        start_time = time.time()
        
        # 1. Retrieval Optimization Agent (SKIPPED for latency)
        # optimization = self.query_rewriter_agent(question)
        queries_to_run = [question]
        optimization = {"rewrite_required": False}
        
        try:
            if not ollama_llm.is_ready():
                return {"success": False, "error": "Ollama LLM not available."}

            # Retrieve & Rank (Passes all queries)
            t_retrieval_start = time.time()
            relevant_chunks, retrieval_stats = self._retrieve_and_rank(
                queries=queries_to_run,
                original_query=question,
                file_id=file_id,
                folder_id=folder_id,
                top_k=max_chunks
            )
            t_retrieval_end = time.time()
            logger.info(f"[TIMER] Retrieval & Reranking (Pass 1): {(t_retrieval_end - t_retrieval_start)*1000:.2f}ms")
            
            # Build Context
            context = self._build_context(relevant_chunks, max_length=8000, folder_id=folder_id)
            
            # Sufficiency Check - Skip if no chunks at all to save an LLM call
            if not relevant_chunks:
                is_sufficient = False
                missing_reason = "No chunks found in retrieval."
            else:
                t_suff_start = time.time()
                is_sufficient, missing_reason = self._check_sufficiency(question, context)
                t_suff_end = time.time()
                logger.info(f"[TIMER] Sufficiency Check: {(t_suff_end - t_suff_start)*1000:.2f}ms")
            
            # 2. Agentic Loop (Pass 2) - ONLY if needed
            if not is_sufficient:
                logger.info(f"Pass 1 Insufficient: {missing_reason}. Reformulating...")
                
                # Reformulate (Fallback to old simple logic or ask agent again?)
                new_query = self._reformulate_query(question, missing_reason)
                
                # Retrieve Pass 2
                chunks_p2, _ = self._retrieve_and_rank(
                    queries=[new_query],
                    original_query=new_query,
                    file_id=file_id,
                    folder_id=folder_id,
                    top_k=max_chunks
                )
                
                # Merge Evidence
                seen_ids = set(c["chunk_id"] for c in relevant_chunks)
                for c in chunks_p2:
                    if c["chunk_id"] not in seen_ids:
                        relevant_chunks.append(c)
                        seen_ids.add(c["chunk_id"])
                
                context = self._build_context(relevant_chunks, max_length=10000, folder_id=folder_id)

            # 3. Final Generation
            if not context.strip() and not folder_id:
                answer = "I'm sorry, I couldn't find any relevant information to answer your question."
                sources = []
            else:
                t_gen_start = time.time()
                # Pass "Antigravity" compliant instructions via system prompt override
                answer = ollama_llm.answer_question(context, question)
                t_gen_end = time.time()
                logger.info(f"[TIMER] Final LLM Generation: {(t_gen_end - t_gen_start)*1000:.2f}ms")
                sources = self._extract_sources(relevant_chunks)

            total_time = (time.time() - start_time) * 1000

            # Audit
            audit_service.log_rag_trace(
                query=question,
                initial_retrieval_count=retrieval_stats.get("initial", 0),
                filtered_count=retrieval_stats.get("filtered", 0),
                reranked_chunks=[], 
                selected_chunks=relevant_chunks,
                context_used=context,
                llm_response=answer,
                generation_time_ms=total_time,
                models_info={
                    "mode": "agentic_loop" if not is_sufficient else "optimized",
                    "rewritten": optimization.get("rewrite_required")
                },
                file_id=file_id,
                folder_id=folder_id
            )

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "chunks_used": len(relevant_chunks),
                "question": question,
                "is_agentic": not is_sufficient
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        chunks: List[Dict],
        max_length: int,
        folder_id: str = None
    ) -> str:
        parts = []
        length = 0

        # 1. Add System Context (Metadata)
        if folder_id:
            folder_info = folder_service.folders.get(folder_id)
            if folder_info:
                folder_name = folder_info.get("name", "Unknown Folder")
                files = folder_service.get_files_in_folder(folder_id)
                if files:
                    # Truncate file list to prevent eating up all context
                    if len(files) > 50:
                        display_files = files[:50]
                        file_names = ", ".join(display_files) + f" ... and {len(files)-50} more."
                    else:
                        file_names = ", ".join(files)
                else:
                    file_names = "No files"
                
                sys_context = f"[System Context]\nCurrent Folder: {folder_name}\nFiles in Folder: {file_names}\n\n"
                parts.append(sys_context)
                length += len(sys_context)

        for i, chunk in enumerate(chunks, 1):
            text = chunk["text"]
            # Previously truncated to 500, now let's allow more per chunk if the total budget allows
            # But we still want to avoid one massive chunk taking all space.
            # Let's cap individual chunks at 1500 chars.
            if len(text) > 1500:
                text = text[:1500] + "..."

            # Resolve Section Name from File Name
            file_id = chunk.get("file_id")
            section_name = "Unknown"
            if file_id in self.document_metadata:
                file_name = self.document_metadata[file_id]["file_name"]
                # Strip extension (e.g. "Report.pdf" -> "Report")
                section_name = Path(file_name).stem
            else:
                # Fallback if metadata missing
                section_name = file_id

            entry = f"[Section: {section_name} | Page {chunk.get('page', '?')}]\n{text}\n"
            
            # Check total length
            if length + len(entry) > max_length:
                break

            parts.append(entry)
            length += len(entry)

        return "\n".join(parts)

    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        sources = []
        for chunk in chunks:
            file_id = chunk.get("file_id")
            if file_id in self.document_metadata:
                meta = self.document_metadata[file_id]
                sources.append({
                    "file_name": meta["file_name"],
                    "page": chunk.get("page"),
                    "chunk_id": chunk.get("chunk_id"),
                })
        return sources

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_processed_document(
        self,
        file_id: str,
        chunks: List[Dict],
        metadata: Dict
    ):
        try:
            output = self.processed_dir / f"{file_id}.json"
            with open(output, "w", encoding="utf-8") as f:
                json.dump(
                    {"metadata": metadata, "chunks": chunks},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.error(f"Error saving processed document: {e}")

    def _load_existing_documents(self):
        try:
            import concurrent.futures
            
            json_files = list(self.processed_dir.glob("*.json"))
            if not json_files:
                return

            logger.info(f"Loading {len(json_files)} processed documents...")
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all load tasks - Load ONLY metadata (chunks=False) to speed up startup
                future_to_file = {executor.submit(self.load_processed_document, f.stem, False): f for f in json_files}
                
                # Wait for completion (optional, but we want them loaded before ready)
                for future in concurrent.futures.as_completed(future_to_file):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error loading document: {e}")
                        
            logger.info("Finished loading processed documents (Metadata Only).")
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")

    def load_processed_document(self, file_id: str, load_chunks: bool = False) -> bool:
        try:
            path = self.processed_dir / f"{file_id}.json"
            if not path.exists():
                return False

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if load_chunks:
                self.document_chunks[file_id] = data["chunks"]
            
            
            self.document_metadata[file_id] = data["metadata"]
            return True

        except Exception as e:
            logger.error(f"Error loading processed document: {e}")
            return False

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def list_documents(self) -> List[Dict]:
        docs = []
        for fid, meta in self.document_metadata.items():
            folder_id = folder_service.get_folder_for_file(fid)
            docs.append({"file_id": fid, "folder_id": folder_id, **meta})
        return docs

    def get_document_info(self, file_id: str) -> Optional[Dict]:
        if file_id in self.document_metadata:
            meta = self.document_metadata[file_id]
            # Use metadata count if available, otherwise fall back to loaded chunks
            count = meta.get("num_chunks", 0)
            if count == 0 and file_id in self.document_chunks:
                count = len(self.document_chunks[file_id])
                
            return {
                "file_id": file_id,
                **meta,
                "chunks_count": count,
            }
        return None

    def _hydrate_chunk(self, file_id: str, chunk_id: str, chunk_index: int = None) -> Optional[Dict]:
        """Load text from memory/disk for a given chunk ID"""
        # 1. Check Memory
        if file_id not in self.document_chunks:
            # Load on demand
             loaded = self.load_processed_document(file_id, load_chunks=True)
             if not loaded:
                 return None
        
        # 2. Find chunk
        chunks = self.document_chunks.get(file_id, [])
        for c in chunks:
            # Helper: handle flexible ID matching
            if str(c.get("chunk_id")) == str(chunk_id):
                return c
        
        # 3. Fallback: by index if payload had it (Rescues mismatched IDs)
        if chunk_index is not None:
             try:
                 # Ensure index is integer
                 c_idx = int(chunk_index)
                 if 0 <= c_idx < len(chunks):
                     # Update the chunk in memory to have the correct ID for future
                     chunks[c_idx]["chunk_id"] = chunk_id
                     return chunks[c_idx]
             except:
                 pass

        return None

    def _retrieve_and_rank(self, queries: List[str], original_query: str, file_id, folder_id, top_k=5) -> Tuple[List[Dict], Dict]:
        # 1. Retrieval (Hybrid delegated to QdrantService)
        # qdrant_service.search now performs Dense + Sparse + Fusion
        
        all_candidates_map = {} 
        search_k = 250
        
        for q in queries:
             results = qdrant_service.search(
                q, 
                k=search_k, 
                folder_id=folder_id,
                file_id=file_id
            )
            
             for res in results:
                cid = res["chunk_id"]
                if cid not in all_candidates_map:
                    # Hydrate
                    payload = res.get("payload", {})
                    fid = payload.get("doc_id") or payload.get("file_id")
                    
                    if fid:
                        c_idx = payload.get("chunk_index")
                        chunk_data = self._hydrate_chunk(fid, cid, chunk_index=c_idx)
                        if chunk_data:
                            # Merge Qdrant Score
                            chunk_data["qdrant_score"] = res["score"]
                            # Add to candidates
                            all_candidates_map[cid] = {
                                "chunk": chunk_data,
                                "score": res["score"],
                                "id": cid
                            }
        
        all_candidates = list(all_candidates_map.values())
        
        # Rerank against ORIGINAL query (Top 20 -> Top 5-8 as per rules, or just all 40?)
        # User said "Rerank top 20 candidates down to top 5-8".
        # If we pulled 40, let's pass top 40 to reranker? Or cut to 20? 
        # Recall > Accuracy implies we should Rerank ALL retrieved if possible. 
        # But let's stick to the prompt hint: "k=20-40" implies retrieval size. "Rerank top 20" implies reranker input.
        # I'll pass all (up to 40) to reranker, and return top_k (default 5 or 8).
        
        rerank_input = all_candidates # No slicing needed if k=40, it's small enough.
        reranked = reranker_service.rerank(original_query, rerank_input, top_k=top_k)
        
        relevant_chunks = [r["chunk"] for r in reranked]
        
        stats = {
            "initial_recall": len(all_candidates),
            "final_count": len(relevant_chunks)
        }
        return relevant_chunks, stats

    def _check_sufficiency(self, query: str, context: str) -> Tuple[bool, str]:
        """
        Always return sufficient to force an answer attempt.
        """
        if not context.strip():
            return False, "No context provided."
            
        return True, "Sufficient"

    def _reformulate_query(self, query: str, missing_reason: str) -> str:
        """
        Reformulate query based on what is missing.
        """
        prompt = (
            f"The original search query '{query}' failed because: {missing_reason}\n"
            f"Generate a NEW, BETTER search query to find the missing information. "
            f"Return ONLY the query string."
        )
        try:
            return ollama_llm.generate_response(prompt).strip().replace('"', '')
        except:
            return query

# Global service instance
qa_service = DocumentQAService()
