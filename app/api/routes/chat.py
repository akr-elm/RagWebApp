from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_ingestion_service
from app.services.ingestion_service import IngestionService
from app.models.schemas import ChatRequest, ChatResponse
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: IngestionService = Depends(get_ingestion_service)
):
    """Chat with the initialized pipeline"""
    logger.info(f"💬 Query: {request.question[:100]}...")
    
    try:
        status = service.get_status()
        if not status.get("ready_for_chat", False):
            raise HTTPException(
                status_code=400,
                detail="Pipeline not ready. Complete setup first."
            )
        
        response, sources = service.query(request.question)
        
        return ChatResponse(
            status="success",
            response=response,
            sources=sources,
            question=request.question
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))