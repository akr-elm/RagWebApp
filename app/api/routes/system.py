from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_ingestion_service, get_config_instance
from app.services.ingestion_service import IngestionService
from app.config import Config
from app.models.schemas import StatusResponse, SuccessResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/system", tags=["system"])

@router.get("/status", response_model=StatusResponse)
async def get_status(
    service: IngestionService = Depends(get_ingestion_service)
):
    """Get current pipeline status"""
    try:
        status = service.get_status()
        return StatusResponse(**status)
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/options")
async def get_available_options(
    config: Config = Depends(get_config_instance)
):
    """Get available configuration options"""
    return {
        "providers": config.available_providers,
        "embedders": config.available_embedders,
        "chunking_strategies": config.chunking_strategy,  # Note: your field name
        "document_types": config.document_types,
        "chunk_size_range": {"min": 100, "max": 2000, "default": 800},
        "chunk_overlap_range": {"min": 0, "max": 500, "default": 100},
        "max_file_size_mb": config.max_file_size_mb,
        "default_settings": {
            "provider": config.default_llm,
            "model": config.default_model
        }
    }

@router.post("/reset", response_model=SuccessResponse)
async def reset_pipeline(
    service: IngestionService = Depends(get_ingestion_service)
):
    """Reset the pipeline to initial state"""
    try:
        success = service.reset()
        if not success:
            raise HTTPException(status_code=500, detail="Reset failed")
        
        return SuccessResponse(message="Pipeline reset successfully")
        
    except Exception as e:
        logger.error(f"❌ Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(
    service: IngestionService = Depends(get_ingestion_service),
    config: Config = Depends(get_config_instance)
):
    """Health check endpoint"""
    try:
        status = service.get_status()
        is_valid = config.validate()
        
        return {
            "status": "healthy" if is_valid else "unhealthy",
            "app_name": config.app_name,
            "version": config.version,
            "config_valid": is_valid,
            "pipeline_ready": status.get("ready_for_chat", False),
            "files_processed": status.get("files_processed", 0)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }