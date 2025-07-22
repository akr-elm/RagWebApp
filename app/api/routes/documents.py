from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import List
from app.api.dependencies import get_ingestion_service, get_config_instance
from app.services.ingestion_service import IngestionService
from app.config import Config
from app.models.schemas import SuccessResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=SuccessResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    service: IngestionService = Depends(get_ingestion_service),
    config: Config = Depends(get_config_instance)
):
    """Upload and process documents"""
    logger.info(f"📤 Uploading {len(files)} files")
    
    # Validate files using your config
    allowed_extensions = {f".{ext}" for ext in config.document_types}
    
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="File must have a filename"
            )
        
        file_ext = f".{file.filename.split('.')[-1].lower()}"
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed types: {list(allowed_extensions)}"
            )
        
        # Check file size (approximate, since UploadFile doesn't have size directly)
        # You might want to implement this check in your service layer
    
    try:
        results = service.process_uploaded_files(files)
        success_count = sum(1 for r in results if r.get("success", False))
        
        if success_count == 0:
            raise HTTPException(
                status_code=400, 
                detail="No files were successfully processed"
            )
        
        return SuccessResponse(
            message=f"Processed {success_count}/{len(files)} files",
            data={
                "results": results, 
                "files_processed": success_count,
                "max_file_size_mb": config.max_file_size_mb,
                "allowed_types": list(allowed_extensions)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))