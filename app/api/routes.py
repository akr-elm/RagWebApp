# app/api/routes.py
from fastapi import APIRouter, UploadFile, HTTPException, Form
from typing import List, Optional
from pydantic import BaseModel
from app.services.ingestion_service import IngestionService

router = APIRouter()

# Global service instance (or use dependency injection)
service = IngestionService()

class PipelineConfig(BaseModel):
    provider: str
    model: str
    embedder: str
    chunking_strategy: Optional[str] = "fixed"
    chunk_size: Optional[int] = 800
    chunk_overlap: Optional[int] = 100

class ChatRequest(BaseModel):
    question: str

@router.get("/available-options")
async def get_available_options():
    """Get available providers, models, and embedders"""
    return service.get_available_options()

@router.post("/upload-documents")
async def upload_documents(files: List[UploadFile]):
    """Step 1: Upload and process documents"""
    try:
        results = service.process_files(files)
        return {
            "status": "success",
            "message": f"Processed {len(files)} files",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/configure-pipeline")
async def configure_pipeline(config: PipelineConfig):
    """Step 2: Configure pipeline with user selections"""
    try:
        success = service.configure_pipeline(
            provider=config.provider,
            model=config.model,
            embedder=config.embedder,
            chunking_strategy=config.chunking_strategy,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        if success:
            return {
                "status": "success",
                "message": "Pipeline configured successfully",
                "configuration": config.dict()
            }
        else:
            raise HTTPException(status_code=400, detail="Configuration failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize-pipeline")
async def initialize_pipeline():
    """Step 3: Initialize the configured pipeline"""
    try:
        success = service.initialize_pipeline()
        
        if success:
            return {
                "status": "success",
                "message": "Pipeline initialized and ready for chat"
            }
        else:
            raise HTTPException(status_code=400, detail="Initialization failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat(request: ChatRequest):
    """Step 4: Chat with the initialized pipeline"""
    try:
        pipeline = service.get_pipeline()
        response, sources = pipeline.query(request.question)
        
        return {
            "status": "success",
            "response": response,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """Get current pipeline status"""
    return service.get_status()

@router.post("/reset")
async def reset_pipeline():
    """Reset the pipeline to initial state"""
    try:
        service.reset()
        return {"status": "success", "message": "Pipeline reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))