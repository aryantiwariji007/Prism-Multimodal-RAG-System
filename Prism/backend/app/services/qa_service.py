"""
Document Q&A Service
Combines document processing with Ollama (DeepSeek) for answering questions
"""

import os
import json
from typing import List, Dict, Optional
from pathlib import Path
import logging
import sys
import time

logger = logging.getLogger(__name__)

# Add the backend directory to the Python path for ingestion modules
backend_dir = Path(__file__).parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# ✅ Fix for [WinError 127] shm.dll: Import torch before PaddleOCR/others
try:
    import torch
except ImportError:
    pass

try:
    from ingestion.parse_pdf import parse_document
    from ingestion.chunker import document_chunker
    from ingestion.image_ingestor import ingest_image
    from ingestion.audio_ingestor import ingest_audio
    from ingestion.video_ingestor import ingest_video
except ImportError as e:
    logger.error(f"Import error for ingestion modules: {e}")
    raise ImportError(
        f"Failed to import ingestion modules. Please ensure dependencies are installed: {e}"
    )

# ✅ Ollama / DeepSeek LLM service
from .llm_service import ollama_llm
from .progress_service import progress_service
# LlamaIndex removed
from .folder_service import folder_service
from .audit_service import audit_service
from .vector_service import vector_service
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
                 chunks = ingest_image(
                     str(file_path),
                     file_id=file_id,
                     progress_callback=report_progress_step
                 )

            elif suffix in {'.mp3', '.wav', '.m4a'}:
                 # Audio Ingestion (Transcribe)
                 chunks = ingest_audio(
                     str(file_path),
                     file_id=file_id,
                     progress_callback=report_progress_step
                 )

            elif suffix in {'.mp4', '.avi', '.mov', '.mkv'}:
                 # Video Ingestion
                 chunks = ingest_video(
                     str(file_path),
                     file_id=file_id,
                     progress_callback=report_progress_step
                 )

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
            
            # Add to Vector Store (FAISS)
            vector_service.add_documents(chunks)

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

    def optimize_query(self, query: str) -> str:
        """
        Refines the query if it is ambiguous or lacks keywords.
        Returns the original query if no optimization is needed.
        """
        # Heuristic: Skip for very short nav queries or very long specific queries
        if len(query.split()) > 10:
            return query 

        prompt = (
            f"Analyze this search query: '{query}'.\n"
            "If it is a specific technical question, code, or distinct name, return 'ORIGINAL'.\n"
            "If it is vague, ambiguous, or conversational (e.g., 'what does it say', 'give me the policy'), "
            "rewrite it to be a standalone, keyword-rich search query for a corporate knowledge base.\n"
            "Do NOT add explanations. Return ONLY the rewritten query or 'ORIGINAL'."
        )
        
        try:
            optimized = ollama_llm.generate_response(prompt).strip()
            # Cleanup
            optimized = optimized.replace('"', '').replace("'", "")
            if "ORIGINAL" in optimized or optimized == query:
                return query
            
            logger.info(f"Query Optimized: '{query}' -> '{optimized}'")
            return optimized
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return query

    def answer_question(
        self,
        question: str,
        file_id: str = None,
        folder_id: str = None,
        max_chunks: int = 6
    ) -> Dict:
        start_time = time.time()
        
        # 0. Query Optimization
        search_query = self.optimize_query(question)
        
        try:
            if not ollama_llm.is_ready():
                return {
                    "success": False,
                    "error": "Ollama LLM not available. Is Ollama running?"
                }

            # Determine search depth (k)
            # If searching in a specific folder/file, we need a much larger K because 
            # the relevant docs might be ranked low globally (e.g. rank 500) but represent
            # the best matches *within* that folder.
            search_k = 5000 if (folder_id or file_id) else 100

            # Retrieve candidates from FAISS using OPTIMIZED query
            candidates = vector_service.search(search_query, k=search_k)
            
            # Filter by folder/file if needed (Client-side filtering for now)
            filtered_candidates = []
            allowed_files = set()
            
            if folder_id:
                folder_files = folder_service.get_files_in_folder(folder_id)
                allowed_files.update(folder_files)
                # Debug logging
                # logger.info(f"Filtering for folder {folder_id} with {len(folder_files)} files.")

            if file_id:
                allowed_files.add(file_id)
                
            # If we have any restrictions, apply them
            if folder_id or file_id:
                for cand in candidates:
                    cand_file_id = cand["chunk"].get("file_id")
                    if cand_file_id in allowed_files:
                        filtered_candidates.append(cand)
            else:
                filtered_candidates = candidates
            
            # Limit for reranker to ensure performance
            # Hybrid Search Strategy: "Keyword Rescue"
                # KEYWORD RESCUE STRATEGY
                # Even if vector search score is low, if the chunk matches specific high-value terms
                # from the query (like "QD040", "Triage"), we rescue it.
                
                # Extract potential "Codes" vs "Common Words"
                # A code is:
                # 1. Alphanumeric mixed (e.g. QD040)
                # 2. Capitalized words (e.g. Triage, Nurse)
                # 3. Hyphenated technical terms (e.g. Safety-Critical)
                import re
                
                # 1. Extract significant terms: Capitals, Alphanumerics, Digits
                # This regex captures: 
                # - [A-Z][a-zA-Z0-9-]* : Capitalized words potentially with numbers/dashes (e.g. Triage, QD-040)
                # - [a-zA-Z]*\d+...    : Words starting with lowercase but containing digits (e.g. v1, 2024-report)
                query_terms_raw = re.findall(r'\b[A-Z][a-zA-Z0-9-]*\b|\b[a-zA-Z]*\d+[a-zA-Z0-9-]*\b', question)
                query_terms = set(query_terms_raw)
                
                # 2. Extract ALL words for strict filename matching (to catch "policy" in "HR Policy.pdf")
                all_query_words = set(re.findall(r'\b[a-zA-Z0-9-]+\b', question.lower()))

                rescued_candidates = []
                other_candidates = []
                
                start_filter_time = time.time()
                
                for cand in filtered_candidates:
                    chunk = cand["chunk"] # This is usually how it's structured in memory before flatten
                    # Depending on how candidates coming from vector_service are structured
                    # vector_service.search returns list of metadata dicts directly mixed?
                    # Let's check vector_service.search output. It returns list of dicts.
                    # so cand is the dict.
                    
                    text = cand.get("text", "").lower()
                    file_id = cand.get("file_id")
                    
                    # 1. Filename Match
                    file_name = ""
                    if file_id in self.document_metadata:
                        file_name = self.document_metadata[file_id].get("file_name", "").lower()
                    
                    filename_match = False
                    if file_name:
                         # Split filename into searchable tokens too
                         file_words = set(re.findall(r'\b[a-zA-Z0-9-]+\b', file_name))
                         
                         common_words = all_query_words.intersection(file_words)
                         # Ignore very common words in filename match
                         common_words = {w for w in common_words if len(w) > 3}
                         if common_words:
                             filename_match = True

                    # 2. Term match (Names, Numbers)
                    text_match = False
                    for term in query_terms:
                        t_lower = term.lower()
                        if len(t_lower) < 3: 
                            continue # Skip very short codes to avoid false positives
                        if t_lower in text:
                            text_match = True
                            break
                    
                    if filename_match or text_match:
                        rescued_candidates.append(cand)
                    else:
                        other_candidates.append(cand)
                
                # Combine: prioritized first, then others
                filtered_candidates = rescued_candidates + other_candidates
                
            # Increased limit to 1000 for better recall (handling folder scopes where relevant docs might be deep)
            if len(filtered_candidates) > 1000:
                filtered_candidates = filtered_candidates[:1000]



                
            # Rerank
            reranked_results = reranker_service.rerank(
                question, 
                filtered_candidates, 
                top_k=max_chunks
            )
            
            # Extract final chunks
            relevant_chunks = [r["chunk"] for r in reranked_results]
            
            # Stats for logging
            initial_count = len(candidates)
            filtered_count = len(filtered_candidates)
            reranked_raw = reranked_results # List with scores

            # Build context (include system context if folder/file selected)
            context = self._build_context(
                relevant_chunks,
                max_length=8000,
                folder_id=folder_id
            )
            
            # Debug: Write context to file
            try:
                with open("debug_context.txt", "w", encoding="utf-8") as f:
                    f.write(context)
            except:
                pass

            # If we have context (chunks OR system context), we try to answer
            if not context.strip():
                answer = "I'm sorry, I couldn't find any relevant information to answer your question."
                sources = []
            else:
                answer = ollama_llm.answer_question(context, question)
                # Only extract sources if we actually used chunks
                sources = self._extract_sources(relevant_chunks) if relevant_chunks else []
            
            total_time = (time.time() - start_time) * 1000

            # Log the full RAG trace (moved up to log even if 0 chunks)
            audit_service.log_rag_trace(
                query=question,
                initial_retrieval_count=initial_count,
                filtered_count=filtered_count,
                reranked_chunks=reranked_raw,  # These have scores
                selected_chunks=relevant_chunks,
                context_used=context,
                llm_response=answer,
                generation_time_ms=total_time,
                models_info={
                    "embedding_model": "nomic-embed-text", 
                    "reranker_model": "ms-marco-Minilm-L-6-v2",
                    "llm_model": os.getenv("TEXT_MODEL_ID", "llama3.2")
                },
                file_id=file_id,
                folder_id=folder_id
            )

            if not context.strip():
                return {
                    "success": False,
                    "error": "No relevant documents found.",
                    "answer": answer
                }

            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "chunks_used": len(relevant_chunks),
                "question": question,
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            msg = f"Error answering question: {e}"
            logger.error(msg)
            # Log error to audit as well so we can see it
            try:
                audit_service.log_event("ERROR", {"error": str(e), "question": question})
            except:
                pass
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
            for json_file in self.processed_dir.glob("*.json"):
                self.load_processed_document(json_file.stem)
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")

    def load_processed_document(self, file_id: str) -> bool:
        try:
            path = self.processed_dir / f"{file_id}.json"
            if not path.exists():
                return False

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

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
            return {
                "file_id": file_id,
                **self.document_metadata[file_id],
                "chunks_count": len(self.document_chunks.get(file_id, [])),
            }
        return None


# Global service instance
qa_service = DocumentQAService()
