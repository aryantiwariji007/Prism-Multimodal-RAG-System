from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
try:
    import torch
except ImportError:
    pass # Handle gracefully if torch is missing entirely


# ... (omitting lines for brevity, the tool finds the import line by context or I replace just the import)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import logging
from typing import Optional
import uuid

from .services.qa_service import qa_service
from .services.audit_service import audit_service
from .services.llm_service import ollama_llm
from .services.progress_service import progress_service
from .services.progress_service import progress_service
from .services.audio_service import audio_service
from .services.folder_service import folder_service
import base64

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# App
# -------------------------------------------------

app = FastAPI(
    title="Prism Document Q&A API",
    version="1.0.0",
)

# -------------------------------------------------
# CORS
# -------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Data directories
# -------------------------------------------------

Path("data/uploads").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/api/uploads", StaticFiles(directory="data/uploads"), name="uploads")

# -------------------------------------------------
# Pydantic models
# -------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    file_id: Optional[str] = None
    folder_id: Optional[str] = None

class QuestionResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    sources: Optional[list] = None
    chunks_used: Optional[int] = None
    error: Optional[str] = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    success: bool
    response: Optional[str] = None
    model_info: Optional[dict] = None
    error: Optional[str] = None

# -------------------------------------------------
# Root / Health
# -------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "Prism Document Q&A API",
        "version": "1.0.0",
        "llm_ready": ollama_llm.is_ready(),
        "model_name": os.getenv("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud"),
        "provider": "ollama",
    }

# -------------------------------------------------
# Model status
# -------------------------------------------------

@app.get("/api/model/status")
async def model_status():
    return {
        "model_loaded": ollama_llm.is_ready(),
        "model_name": os.getenv("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud"),
        "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "provider": "ollama",
    }

# -------------------------------------------------
# Audio placeholder
# -------------------------------------------------

@app.get("/api/audio")
async def list_audio():
    docs = qa_service.list_documents()
    audio_files = []
    # Base URL for static files
    # TODO: In production, this should be properly configured
    
    for doc in docs:
        if doc.get('type') in {'mp3', 'wav', 'm4a'}:
             audio_files.append({
                "file_id": doc['file_id'],
                "file_name": doc['file_name'],
                "folder_id": doc.get("folder_id"),
                "size": 0, # TODO: Store size in metadata
                "url": f"/api/uploads/{doc['file_name']}",
                "duration": 0
            })
            
    return {
        "success": True,
        "audio": audio_files
    }

@app.get("/api/images")
async def list_images():
    docs = qa_service.list_documents()
    image_files = []
    
    for doc in docs:
        if doc.get('type') in {'jpg', 'jpeg', 'png', 'webp'}:
             image_files.append({
                "file_id": doc['file_id'],
                "file_name": doc['file_name'],
                "folder_id": doc.get("folder_id"),
                "size": 0,
                "url": f"/api/uploads/{doc['file_name']}"
            })

    return {
        "success": True,
        "images": image_files
    }

@app.get("/api/videos")
async def list_videos():
    docs = qa_service.list_documents()
    video_files = []
    
    for doc in docs:
        if doc.get('type') in {'mp4', 'avi', 'mov', 'mkv'}:
             video_files.append({
                "file_id": doc['file_id'],
                "file_name": doc['file_name'],
                "folder_id": doc.get("folder_id"),
                "size": 0,
                "url": f"/api/uploads/{doc['file_name']}"
            })

    return {
        "success": True,
        "videos": video_files
    }

class ImageQuestionRequest(BaseModel):
    question: str
    image_id: str

@app.post("/api/image-question")
async def ask_image_question(request: ImageQuestionRequest):
    # Find the image file
    upload_dir = Path("data/uploads")
    image_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        possible_path = upload_dir / f"{request.image_id}{ext}"
        if possible_path.exists():
            image_path = possible_path
            break
    
    if not image_path:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        # Read image and convert to base64
        # Read image and convert to base64, ensuring compatible format (JPEG)
        import io
        from PIL import Image
        
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            
        # Convert to JPEG using PIL
        # access to io and Image must be ensured (can be moved to top imports later or here locally)
        try:
            image = Image.open(io.BytesIO(img_bytes))
            # Convert RGBA/P to RGB for JPEG
            if image.mode in ('RGBA', 'P', 'LA'):
                image = image.convert('RGB')
                
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=95)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as img_err:
            logger.warning(f"Failed to convert image {image_path}, falling back to raw bytes: {img_err}")
            image_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        response = ollama_llm.generate_vision_response(
            prompt=request.question,
            image_base64=image_base64
        )
        
        sources = [{"file_name": image_path.name}]

        # Log to audit history
        audit_service.log_event("IMAGE_TRACE", {
            "question": request.question,
            "answer": response,
            "sources": sources,
            "image_id": request.image_id
        })
        
        return {
            "success": True,
            "answer": response,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Image Q&A error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

class AudioQuestionRequest(BaseModel):
    question: str
    audio_id: str

@app.post("/api/audio-question")
async def ask_audio_question(request: AudioQuestionRequest):
    # Find the audio file
    upload_dir = Path("data/uploads")
    audio_path = None
    for ext in ['.mp3', '.wav', '.m4a']:
        possible_path = upload_dir / f"{request.audio_id}{ext}"
        if possible_path.exists():
            audio_path = possible_path
            break
            
    if not audio_path:
        raise HTTPException(status_code=404, detail="Audio not found")

    try:
        # Check for existing transcript
        transcript_path = audio_path.with_suffix('.txt')
        transcript = ""
        
        if transcript_path.exists():
            transcript = transcript_path.read_text(encoding='utf-8')
        else:
            # Transcribe
            transcript = audio_service.transcribe(str(audio_path))
            # Save for future use
            transcript_path.write_text(transcript, encoding='utf-8')
            
        # Ask LLM
        context = f"Audio Transcript:\n{transcript}"
        answer = ollama_llm.answer_question(context, request.question)
        
        sources = [{"file_name": audio_path.name, "timestamp": "Full Audio"}]

        # Log to audit history
        audit_service.log_event("AUDIO_TRACE", {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "audio_id": request.audio_id
        })
        
        return {
            "success": True,
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Audio Q&A error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

class VideoQuestionRequest(BaseModel):
    question: str
    video_id: str

@app.post("/api/video-question")
async def ask_video_question(request: VideoQuestionRequest):
    # Use the general QA service tailored to this file
    result = qa_service.answer_question(
        question=request.question,
        file_id=request.video_id
    )

    if result["success"]:
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    
    return {
        "success": False,
        "error": result["error"]
    }

# -------------------------------------------------
# Processing status (FIXED)
# -------------------------------------------------

from .services.ingestion_service import ingestion_service

@app.on_event("startup")
async def startup_event():
    await ingestion_service.start()

@app.on_event("shutdown")
async def shutdown_event():
    await ingestion_service.stop()

# -------------------------------------------------
# Processing status
# -------------------------------------------------

@app.get("/api/processing-status/{processing_id}")
async def get_processing_status(processing_id: str):
    # Check IngestionService first (Persistent)
    status = ingestion_service.get_status(processing_id)
    if status:
        return {
            "success": True,
            "file_id": processing_id,
            "status": status["status"],
            "percentage": 50 if status["current_step_number"] == 1 else 100, # Approximate
            "message": status["current_step"],
            "current_step_number": status["current_step_number"],
            "total_steps": status["total_steps"],
            "error_message": status["error_message"],
        }

    # Fallback to legacy progress service (In-Memory)
    progress = progress_service.get_progress(processing_id)

    if not progress:
        return {
            "success": True,
            "file_id": processing_id,
            "status": "pending",
            "percentage": 0,
            "message": "Processing not started yet",
            "current_step_number": 0,
            "total_steps": 2,
            "estimated_remaining": None,
            "error_message": None,
        }

    return {
        "success": True,
        "file_id": progress.file_id,
        "status": progress.status,
        "percentage": progress.progress,
        "message": progress.current_step,
        "current_step_number": progress.current_step_number,
        "total_steps": progress.total_steps,
        "estimated_remaining": progress.estimated_remaining,
        "error_message": progress.error_message,
    }

# -------------------------------------------------
# Upload document
# -------------------------------------------------

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    folder_id: Optional[str] = Form(None),
):
    allowed_extensions = {
        ".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".ppt",
        ".jpg", ".jpeg", ".png", ".webp", 
        ".mp3", ".wav", ".m4a", 
        ".mp4", ".avi", ".mov", ".mkv",
        ".xlsx", ".xls", ".csv", ".json", ".xml", ".html", ".htm"
    }
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    # Use UUID for internal storage
    safe_filename = f"{uuid.uuid4()}_{Path(file.filename).name}"
    file_id = safe_filename
    
    upload_path = Path("data/uploads") / safe_filename

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Assign to folder if requested
    if folder_id:
        try:
            folder_service.assign_file(file_id, folder_id)
        except Exception as e:
            logger.error(f"Failed to assign file {file_id} to folder {folder_id}: {e}")

    # Use IngestionService (Async & Persistent)
    ingestion_service.add_job(str(upload_path), file_id, folder_id)

    return {
        "success": True,
        "file_id": file_id,
        "file_name": file.filename,  
        "progress_id": file_id, # processing_id is same as file_id now
        "status_url": f"/api/processing-status/{file_id}",
    }

# -------------------------------------------------
# Chat
# -------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not ollama_llm.is_ready():
        raise HTTPException(status_code=503, detail="Ollama server not running")

    response_text = ollama_llm.generate_response(
        prompt=request.message,
        max_tokens=512,
        temperature=0.7,
    )

    # Log to audit history
    audit_service.log_event("CHAT_TRACE", {
        "message": request.message,
        "response": response_text,
        "model": os.getenv("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud")
    })

    return ChatResponse(
        success=True,
        response=response_text,
        model_info={
            "model": os.getenv("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud"),
            "provider": "ollama",
        },
    )

# -------------------------------------------------
# Document Q&A
# -------------------------------------------------

@app.post("/api/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    result = qa_service.answer_question(
        question=request.question,
        file_id=request.file_id,
        folder_id=request.folder_id,
    )

    if result["success"]:
        return QuestionResponse(
            success=True,
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
        )

    return QuestionResponse(
        success=False,
        error=result["error"],
    )

# -------------------------------------------------
# History / Audit
# -------------------------------------------------

@app.get("/api/history")
async def get_search_history(limit: int = 50):
    """
    Get backend-side audit history for RAG (Search)
    """
    history = audit_service.get_history(limit)
    return {
        "success": True, 
        "history": history
    }

# -------------------------------------------------
# Documents
# -------------------------------------------------

@app.get("/api/documents")
async def list_documents():
    docs = qa_service.list_documents()
    return {
        "success": True,
        "documents": docs,
        "count": len(docs),
    }

@app.get("/api/documents/{file_id}")
async def get_document_info(file_id: str):
    doc = qa_service.get_document_info(file_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True, "document": doc}

@app.delete("/api/documents/{file_id}")
async def delete_document(file_id: str):
    qa_service.document_chunks.pop(file_id, None)
    qa_service.document_metadata.pop(file_id, None)

    processed_file = Path("data/processed") / f"{file_id}.json"
    if processed_file.exists():
        processed_file.unlink()

    return {
        "success": True,
        "message": f"Document '{file_id}' deleted successfully",
    }

# -------------------------------------------------
# Folders
# -------------------------------------------------

class CreateFolderRequest(BaseModel):
    name: str

@app.post("/api/folders")
async def create_folder(request: CreateFolderRequest):
    folder = folder_service.create_folder(request.name)
    return {"success": True, "folder": folder}

@app.get("/api/folders")
async def list_folders():
    folders = folder_service.list_folders()
    return {"success": True, "folders": folders}

@app.delete("/api/folders/{folder_id}")
async def delete_folder(folder_id: str):
    folder_service.delete_folder(folder_id)
    return {"success": True, "message": "Folder deleted"}

class RenameFolderRequest(BaseModel):
    name: str

@app.put("/api/folders/{folder_id}")
async def rename_folder(folder_id: str, request: RenameFolderRequest):
    folder = folder_service.rename_folder(folder_id, request.name)
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    return {"success": True, "folder": folder}

class AssignFileRequest(BaseModel):
    file_id: str

@app.post("/api/folders/{folder_id}/files")
async def assign_file(folder_id: str, request: AssignFileRequest):
    try:
        folder_service.assign_file(request.file_id, folder_id)
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/folders/{folder_id}/files/{file_id}")
async def unassign_file(folder_id: str, file_id: str):
    folder_service.unassign_file(file_id)
    return {"success": True}

class BulkDeleteRequest(BaseModel):
    file_ids: list[str]

@app.delete("/api/files/bulk-delete")
async def bulk_delete_files(request: BulkDeleteRequest):
    deleted_count = 0
    errors = []

    for file_id in request.file_ids:
        try:
            # 1. Remove from memory chunks
            qa_service.document_chunks.pop(file_id, None)
            qa_service.document_metadata.pop(file_id, None)

            # 2. Remove processed JSON
            processed_file = Path("data/processed") / f"{file_id}.json"
            if processed_file.exists():
                processed_file.unlink()

            # 3. Remove physical file (finding by prefix if necessary, but we store full name in metadata usually)
            # However, uploads are stored as uuid_original.name. 
            # We need to find the file in uploads dir that starts with file_id or matches the stored path
            # Ideally we check metadata for the path.
            # If metadata is already popped, we might lose the path, so let's get it first or just glob.
            # file_id in main.py upload is `safe_filename` which IS the filename on disk.
            upload_path = Path("data/uploads") / file_id
            if not upload_path.exists():
                # Try finding by glob if extension is missing/unknown (though file_id usually has extension in this codebase? 
                # Wait, upload says: `file_id = safe_filename`. So likely it includes extension.
                # Just in case, let's try to remove it directly.
                pass
            
            if upload_path.exists():
                 upload_path.unlink()

            # 4. Unassign from folders
            folder_service.unassign_file(file_id)

            # 5. Remove from Vector Store
            llama_service.delete_file(file_id)
            
            deleted_count += 1
            
        except Exception as e:
            errors.append(f"Failed to delete {file_id}: {str(e)}")
            logger.error(f"Bulk delete error for {file_id}: {e}")

    return {
        "success": True, 
        "deleted_count": deleted_count,
        "errors": errors
    }

