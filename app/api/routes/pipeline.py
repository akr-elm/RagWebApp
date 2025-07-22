from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_ingestion_service, get_config_instance
from app.services.ingestion_service import IngestionService
from app.config import Config
from app.models.schemas import PipelineConfig, SuccessResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])

@router.post("/configure", response_model=SuccessResponse)
async def configure_pipeline(
    config_data: PipelineConfig,
    service: IngestionService = Depends(get_ingestion_service),
    config: Config = Depends(get_config_instance)
):
    """Configure pipeline with user selections"""
    logger.info(f"⚙️ Configuring pipeline: {config_data.provider}/{config_data.model}")
    
    # Validate configuration
    _validate_config(config_data, config)
    
    try:
        success = service.configure_pipeline(
            provider=config_data.provider,
            model=config_data.model,
            embedder=config_data.embedder,
            chunking_strategy=config_data.chunking_strategy,
            chunk_size=config_data.chunk_size,
            chunk_overlap=config_data.chunk_overlap
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Configuration failed")
        
        return SuccessResponse(
            message="Pipeline configured successfully",
            data=config_data.dict()
        )
        
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize", response_model=SuccessResponse)
async def initialize_pipeline(
    service: IngestionService = Depends(get_ingestion_service)
):
    """Initialize the configured pipeline"""
    logger.info("🔄 Initializing pipeline...")
    
    try:
        status = service.get_status()
        
        # Check if configuration exists
        if not status.get("configuration", {}).get("provider"):
            raise HTTPException(
                status_code=400,
                detail="Pipeline not configured"
            )
        
        if status.get("files_processed", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No files processed"
            )
        
        success = service.initialize_pipeline()
        if not success:
            raise HTTPException(status_code=400, detail="Initialization failed")
        
        return SuccessResponse(message="Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _validate_config(config_data: PipelineConfig, config: Config):
    """Validate pipeline configuration using your dataclass config"""
    if config_data.provider not in config.available_providers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider: {config_data.provider}. Available: {list(config.available_providers.keys())}"
        )
    
    if config_data.model not in config.available_providers[config_data.provider]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {config_data.model}. Available for {config_data.provider}: {config.available_providers[config_data.provider]}"
        )
    
    if config_data.embedder not in config.available_embedders:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid embedder: {config_data.embedder}. Available: {config.available_embedders}"
        )
    
    if config_data.chunking_strategy not in config.chunking_strategy:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid chunking strategy: {config_data.chunking_strategy}. Available: {config.chunking_strategy}"
        )