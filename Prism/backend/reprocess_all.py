import sys
import asyncio
import logging
from pathlib import Path
import shutil

# Setup environment
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
load_dotenv(backend_dir / ".env")

from app.services.qa_service import qa_service
from app.services.ingestion_service import ingestion_service
from app.services.qdrant_service import qdrant_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reprocess_all():
    print("--- Starting Full Reprocess (Re-Parsing & Re-Indexing) ---")
    
    # 1. Get List of Existing Docs
    docs = qa_service.list_documents()
    print(f"Found {len(docs)} documents to reprocess.")
    
    # 2. Reset Vector DB
    print("Clearing Vector Database...")
    qdrant_service.delete_collection()
    
    # 3. Reprocess Each File
    for doc in docs:
        file_id = doc['file_id']
        file_path = doc.get('file_path')
        
        if not file_path:
            print(f"Skipping {file_id}: No file path found.")
            continue
            
        print(f"Reprocessing: {doc.get('file_name')} ({file_id})...")
        
        # Manually create a job object to mimic the queue processing
        # We call the internal _process_job directly to avoid queue complexity
        # But we need to ensure the DB job record exists or is updated
        
        # Re-add job to Ingestion DB to ensure status tracking works
        ingestion_service.add_job(file_path, file_id, doc.get('folder_id'))
        
        # Run processing directly
        await ingestion_service._process_job(file_id)
        
        print(f"Completed: {doc.get('file_name')}")
        
    print("\n--- Reprocessing Complete ---")
    print("Please restart the backend server to load the new data.")

if __name__ == "__main__":
    asyncio.run(reprocess_all())
