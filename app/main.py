from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import get_config
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
from pydantic import BaseModel
from app.services.ingestion_service import IngestionService

# Get configuration
config = get_config()

# Global service instance
service = IngestionService()

# Pydantic models for request bodies
class PipelineConfig(BaseModel):
    provider: str
    model: str
    embedder: str
    chunking_strategy: str = "fixed"
    chunk_size: int = 800
    chunk_overlap: int = 100

class ChatRequest(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if not config.validate():
        raise Exception("Configuration validation failed")
    print("✅ Configuration validated successfully")
    
    yield  # Application runs here
    
    # Shutdown (if you need cleanup)
    print("🔄 Shutting down application")

app = FastAPI(
    title=config.app_name,
    description="FastAPI backend for your Retrieval-Augmented Generation pipeline",
    version=config.version,
    lifespan=lifespan
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="."), name="static")

# Serve the test interface at root
@app.get("/")
async def serve_test_interface():
    return FileResponse("testing_interface.html")

# Allow CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints
@app.get("/available-options")
async def get_available_options():
    """Get available providers, models, and embedders"""
    return {
        "providers": config.available_providers,
        "embedders": config.available_embedders,
        "chunking_strategies": ["fixed", "semantic", "recursive"]
    }

@app.post("/upload-documents")
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

@app.post("/configure-pipeline")
async def configure_pipeline(config_data: PipelineConfig):
    """Step 2: Configure pipeline with user selections"""
    try:
        success = service.configure_pipeline(
            provider=config_data.provider,
            model=config_data.model,
            embedder=config_data.embedder,
            chunking_strategy=config_data.chunking_strategy,
            chunk_size=config_data.chunk_size,
            chunk_overlap=config_data.chunk_overlap
        )
        
        if success:
            return {
                "status": "success",
                "message": "Pipeline configured successfully",
                "configuration": config_data.dict()
            }
        else:
            raise HTTPException(status_code=400, detail="Configuration failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize-pipeline")
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

@app.post("/chat")
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

@app.get("/status")
async def get_status():
    """Get current pipeline status"""
    return service.get_status()

@app.post("/reset")
async def reset_pipeline():
    """Reset the pipeline to initial state"""
    try:
        service.reset()
        return {"status": "success", "message": "Pipeline reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Register additional routes (if any)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)