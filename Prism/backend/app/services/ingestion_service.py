import asyncio
import sqlite3
import json
import logging
import uuid
import os
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List

# Imports from existing modules
from ..services.qdrant_service import qdrant_service
from ..services.audit_service import audit_service
from ingestion.parse_pdf import parse_document
from ingestion.chunker import document_chunker
# We'll import specific ingestors conditionally or lazily

logger = logging.getLogger(__name__)

class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING_STAGE_1 = "processing_stage_1" # Text / Fast
    PROCESSING_STAGE_2 = "processing_stage_2" # Enrichment / Slow
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class IngestionJob:
    def __init__(self, file_id, file_path, folder_id, status=IngestionStatus.PENDING, error=None):
        self.file_id = file_id
        self.file_path = file_path
        self.folder_id = folder_id
        self.status = status
        self.error = error

class IngestionService:
    def __init__(self, db_path="data/ingestion.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._queue = asyncio.PriorityQueue()
        self._running = False
        self._worker_task = None
        self._loop = None

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    file_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    folder_id TEXT,
                    status TEXT DEFAULT 'pending',
                    stage INTEGER DEFAULT 0,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    async def start(self):
        """Start the background worker"""
        if self._running:
            return
        
        self._running = True
        # Load pending jobs from DB
        self._load_pending_jobs()
        
        self._loop = asyncio.get_event_loop()
        self._worker_task = self._loop.create_task(self._worker_loop())
        logger.info("Ingestion Service started.")

    async def stop(self):
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Ingestion Service stopped.")

    def _load_pending_jobs(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT file_id, file_path, folder_id, status FROM jobs WHERE status NOT IN ('completed', 'failed')"
            )
            for row in cursor:
                # Priority: PARTIAL > PENDING
                priority = 1 if row[3] == IngestionStatus.PARTIAL else 2
                self._queue.put_nowait((priority, row[0])) # Store file_id

    def add_job(self, file_path: str, file_id: str, folder_id: str = None):
        """Register a new ingestion job"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO jobs (file_id, file_path, folder_id, status, stage, updated_at) VALUES (?, ?, ?, ?, 0, CURRENT_TIMESTAMP)",
                    (file_id, str(file_path), folder_id, IngestionStatus.PENDING)
                )
                conn.commit()
            
            self._queue.put_nowait((2, file_id)) # Priority 2 (Normal)
            logger.info(f"Job added: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add job {file_id}: {e}")
            return False

    def get_status(self, file_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT status, stage, error FROM jobs WHERE file_id = ?", (file_id,)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "status": row[0],
                    "current_step": "Processing" if "processing" in row[0] else row[0],
                    "current_step_number": row[1],
                    "total_steps": 2,
                    "error_message": row[2]
                }
        return None

    async def _worker_loop(self):
        logger.info("Worker loop started.")
        while self._running:
            try:
                priority, file_id = await self._queue.get()
                logger.info(f"Picking up job {file_id} (Priority {priority})")
                
                await self._process_job(file_id)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)

    async def _process_job(self, file_id: str):
        # 1. Fetch Job Details
        job = None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT file_path, folder_id, status FROM jobs WHERE file_id = ?", (file_id,))
            row = cursor.fetchone()
            if row:
                job = IngestionJob(file_id, row[0], row[1], row[2])

        if not job:
            return

        try:
            # Update Status -> Processing Stage 1
            self._update_status(file_id, IngestionStatus.PROCESSING_STAGE_1, 1)

            # --- STAGE 1: Text Extraction & Indexing (Blocking CPU task run in thread pool) ---
            orig_path_str = str(job.file_path)
            file_path = Path(orig_path_str)
            
            if not file_path.exists():
                # 1. Try stripping 'backend/' if we are ALREADY in backend
                if orig_path_str.startswith("backend/"):
                    stripped = orig_path_str.replace("backend/", "", 1)
                    if Path(stripped).exists():
                        file_path = Path(stripped)
                
                # 2. Try ADDING 'backend/' if we are in root
                if not file_path.exists():
                    added = "backend/" + orig_path_str
                    if Path(added).exists():
                        file_path = Path(added)
                
                # 3. Try just finding the filename in ANY known uploads dir
                if not file_path.exists():
                    filename = Path(orig_path_str).name
                    # Try data/uploads and backend/data/uploads
                    possibilities = [
                        Path("data/uploads") / filename,
                        Path("backend/data/uploads") / filename
                    ]
                    for p in possibilities:
                        if p.exists():
                            file_path = p
                            break
            
            if not file_path.exists():
                raise FileNotFoundError(f"Could not find document file: {orig_path_str}")

            # Run Stage 1
            chunks = await asyncio.to_thread(self._run_stage_1, file_path, file_id)
            
            metadata = None

            # Commit Stage 1 to Vector DB
            # Note: We do this even if Stage 1 yields partial results
            if chunks:
                await asyncio.to_thread(qdrant_service.add_documents, chunks)
                logger.info(f"Stage 1 Complete for {file_id}: {len(chunks)} text chunks indexed in Qdrant.")
                
                # --- SAVE METADATA (Critical for System Visibility) ---
                # Update QA Service in-memory and persistent JSON
                from ..services.qa_service import qa_service
                
                metadata = {
                    "file_id": file_id,
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "type": file_path.suffix.replace('.', ''),
                    "num_chunks": len(chunks),
                    "total_characters": sum(len(c.get("text", "")) for c in chunks),
                    "ingestion_status": "partial"
                }
                
                # We need to run this thread-safe or assuming Global Interpreter Lock helps us for simple dict assignment
                # Since we are in an async method, we should be careful. 
                # But qa_service is just dicts.
                qa_service.document_chunks[file_id] = chunks
                qa_service.document_metadata[file_id] = metadata
                

                # Save JSON
                await asyncio.to_thread(
                    qa_service._save_processed_document, 
                    file_id, 
                    chunks, 
                    metadata
                )

            # Check if Stage 2 is needed (Images, tables, complex PDF)
            needs_stage_2 = self._check_needs_stage_2(file_path, chunks)
            
            if needs_stage_2:
                self._update_status(file_id, IngestionStatus.PROCESSING_STAGE_2, 2)
                # Re-queue for Stage 2 if we wanted to yield, but here we just continue
                # Run Stage 2
                enrichment_chunks = await asyncio.to_thread(self._run_stage_2, file_path, file_id)
                if enrichment_chunks:
                     await asyncio.to_thread(qdrant_service.add_documents, enrichment_chunks)
                     logger.info(f"Stage 2 Complete for {file_id}: {len(enrichment_chunks)} enrichment chunks indexed in Qdrant.")
                     
                     # Update Metadata with new chunks
                     all_chunks = chunks + enrichment_chunks
                     
                     if metadata is None:
                         metadata = {
                            "file_id": file_id,
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "type": file_path.suffix.replace('.', ''),
                            "ingestion_status": "partial"
                        }

                     metadata["num_chunks"] = len(all_chunks)
                     metadata["total_characters"] = sum(len(c.get("text", "")) for c in all_chunks)
                     metadata["ingestion_status"] = "completed"
                     
                     # Check if qa_service uses lazy loading, but here we push data directly
                     from ..services.qa_service import qa_service
                     qa_service.document_chunks[file_id] = all_chunks
                     qa_service.document_metadata[file_id] = metadata
                     
                     await asyncio.to_thread(
                        qa_service._save_processed_document, 
                        file_id, 
                        all_chunks, 
                        metadata
                    )

            # Complete
            self._update_status(file_id, IngestionStatus.COMPLETED, 2)
            
            # Audit
            audit_service.log_event("INGESTION_COMPLETE", {
                "file_id": file_id,
                "path": str(file_path),
                "chunks": len(chunks)
            })

        except Exception as e:
            logger.exception(f"Job {file_id} failed")
            self._update_status(file_id, IngestionStatus.FAILED, 0, str(e))

    def _update_status(self, file_id, status, stage, error=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, error = ?, updated_at = CURRENT_TIMESTAMP WHERE file_id = ?",
                (status, stage, error, file_id)
            )
            conn.commit()

    def _run_stage_1(self, file_path: Path, file_id: str) -> List[Dict]:
        """
        Fast Text Extraction with Context Injection
        """
        suffix = file_path.suffix.lower()
        chunks = []
        
        if suffix in {'.pdf', '.docx'}:
            chunks = parse_document(str(file_path), file_id, file_id)
        elif suffix == '.pptx':
            from ingestion.pptx_ingestor import ingest_pptx
            chunks = ingest_pptx(str(file_path), file_id=file_id)
        elif suffix in {'.xlsx', '.xls', '.csv'}:
            from ingestion.excel_ingestor import ingest_excel
            chunks = ingest_excel(str(file_path), file_id=file_id)
        elif suffix in {'.txt', '.md', '.json', '.xml', '.html'}:
             text = file_path.read_text(encoding='utf-8', errors='replace')
             chunks = document_chunker.chunk_document_pages([{"text": text, "page": 1, "file_id": file_id}], file_name=file_path.name)
        
        # Mandatory Metadata Enrichment & Validation
        for i, chunk in enumerate(chunks):
            # 1. Update file_type from actual extension
            chunk["file_type"] = suffix.replace('.', '')
            chunk["doc_id"] = file_id
            chunk["source_file"] = file_path.name
            chunk["ingestion_method"] = "text_extraction_fast"
            chunk["chunk_index"] = i
            chunk["content_type"] = chunk.get("type", "text")
            
            # Ensure metadata dict also has these if needed for downstream (consistency)
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            for k in ["doc_id", "source_file", "file_type", "ingestion_method", "chunk_index", "content_type"]:
                chunk["metadata"][k] = chunk[k]
            
            # Context is already injected in chunker.py's _create_chunk
            
        return chunks

    def _check_needs_stage_2(self, file_path: Path, chunks: List[Dict]) -> bool:
        """Heuristic: Do we need deep analysis?"""
        suffix = file_path.suffix.lower()
        if suffix in {'.jpg', '.png', '.jpeg', '.mp3', '.mp4'}:
            return True
        
        # For PDF, if text was sparse, maybe we need OCR?
        if suffix == '.pdf':
            total_text = sum(len(c['text']) for c in chunks)
            # Increased threshold: If a document has less than 1000 chars of text, 
            # it's likely a scan or form with just headers. Force OCR.
            if total_text < 1000: 
                logger.info(f"PDF has low text count ({total_text} chars). Triggering Stage 2 OCR.")
                return True
        
        return False

    def _run_stage_2(self, file_path: Path, file_id: str) -> List[Dict]:
        """Enrichment: Vision, Audio, OCR"""
        chunks = []
        suffix = file_path.suffix.lower()

        try:
            if suffix in {'.jpg', '.png', '.jpeg', '.webp'}:
                from ingestion.image_ingestor import ingest_image
                chunks = ingest_image(str(file_path), file_id=file_id)
                
            elif suffix in {'.mp3', '.wav', '.m4a'}:
                from ingestion.audio_ingestor import ingest_audio
                chunks = ingest_audio(str(file_path), file_id=file_id)
                
            elif suffix == '.pdf':
                # Deep Analysis for PDF (OCR for scanned pages)
                logger.info(f"Starting Stage 2 OCR for PDF: {file_id}")
                chunks = self._run_pdf_ocr(file_path, file_id)

        except Exception as e:
            logger.error(f"Stage 2 failed for {file_id}: {e}")
        
        # Enrich enhancement chunks with stage info
        for i, chunk in enumerate(chunks):
            chunk['ingestion_stage'] = 'enrichment_layer'
            
            # Inject strict metadata (top-level)
            chunk["file_type"] = suffix.replace('.', '')
            chunk["doc_id"] = file_id
            chunk["source_file"] = file_path.name
            chunk["ingestion_method"] = "ocr_enrichment_paddle" if suffix == '.pdf' else "multimodal_model"
            chunk["chunk_index"] = 999 + i # Use distinct offset for stage 2
            chunk["content_type"] = chunk.get("type", "text")
            
            # Ensure metadata dict also has these
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            for k in ["doc_id", "source_file", "file_type", "ingestion_method", "chunk_index", "content_type"]:
                chunk["metadata"][k] = chunk[k]

            chunk["metadata"]["start_time"] = datetime.utcnow().isoformat()
            chunk["metadata"]["ocr_confidence"] = 0.95 

            # Prepend filename if not already done
            original_text = chunk.get("text", "")
            if not original_text.startswith(f"Filename:"):
                chunk["text"] = f"Filename: {file_path.name}\nFile ID: {file_id}\n{original_text}"

        return chunks

    def _run_pdf_ocr(self, file_path: Path, file_id: str) -> List[Dict]:
        """Convert PDF pages to images and run OCR (with LLaVA fallback)"""
        ocr_chunks = []
        try:
            import pdfplumber
            from ingestion.ocr_image import prism_ocr
            from ingestion.chunker import document_chunker
            from ..services.llm_service import llm_service
            
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    logger.info(f"OCR'ing PDF page {i+1}/{len(pdf.pages)}")
                    
                    im = page.to_image(resolution=200) # Slightly lower res for LLM speed
                    temp_img_path = f"data/temp_ocr_{file_id}_p{i}.png"
                    im.save(temp_img_path, format="PNG")
                    
                    # 1. Try Paddle OCR
                    page_text = ""
                    try:
                         page_text = prism_ocr.extract_text(temp_img_path)
                    except Exception as ocr_e:
                         logger.warning(f"PaddleOCR failed, trying LLaVA fallback: {ocr_e}")
                    
                    # 2. Fallback to LLaVA Vision if Paddle failed or returned nothing
                    if not page_text.strip():
                        try:
                            # Use a specific OCR prompt for the vision model
                            ocr_prompt = "Transcribe all the text you see in this image exactly, including names and numbers. Return only the transcription."
                            page_text = llm_service.analyze_image(temp_img_path, ocr_prompt)
                            logger.info(f"LLaVA OCR successful for page {i+1}")
                        except Exception as llm_e:
                            logger.error(f"LLaVA fallback failed: {llm_e}")
                    
                    # 3. Clean up
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                    
                    if page_text.strip():
                        # Chunk the OCR text
                        page_chunks = document_chunker.chunk_text(page_text, file_id, file_name=file_path.name)
                        for c in page_chunks:
                            c["page"] = i + 1
                        ocr_chunks.extend(page_chunks)
                        
        except Exception as e:
            logger.error(f"PDF OCR failed for {file_id}: {e}")
            
        return ocr_chunks

# Global Instance
ingestion_service = IngestionService()
