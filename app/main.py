from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
from pydantic import BaseModel
from app.services.ingestion_service import IngestionService
from app.config import get_config
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get configuration
config = get_config()

# Global service instance
service = IngestionService()

# Pydantic models for request bodies
class PipelineConfig(BaseModel):
    provider: str
    model: str
    embedder: str
    chunking_strategy: str = "langchain_recursive"  # Updated default
    chunk_size: int = 800
    chunk_overlap: int = 100

class ChatRequest(BaseModel):
    question: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        if not config.validate():
            raise Exception("Configuration validation failed")
        logger.info("✅ Configuration validated successfully")
        logger.info(f"🚀 Starting {config.app_name} v{config.version}")
        logger.info(f"📊 Available providers: {list(config.available_providers.keys())}")
        logger.info(f"🧠 Available embedders: {len(config.available_embedders)}")
        logger.info(f"🔪 Available chunking strategies: {config.chunking_strategy}")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("🔄 Shutting down application")

app = FastAPI(
    title=config.app_name,
    description="FastAPI backend for your Retrieval-Augmented Generation pipeline with multiple chunking strategies",
    version=config.version,
    lifespan=lifespan
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="."), name="static")

# Serve the test interface at root
@app.get("/")
async def serve_test_interface():
    """Serve the testing interface"""
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
    """Get available providers, models, embedders, and chunking strategies"""
    try:
        return {
            "providers": config.available_providers,
            "embedders": config.available_embedders,
            "chunking_strategies": config.chunking_strategy,
            "chunk_size_range": {"min": 100, "max": 2000, "default": 800},
            "chunk_overlap_range": {"min": 0, "max": 500, "default": 100}
        }
    except Exception as e:
        logger.error(f"Error getting available options: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Step 1: Upload and process documents"""
    try:
        logger.info(f"📤 Uploading {len(files)} files")
        
        # Validate file types
        allowed_types = {'.txt', '.pdf', '.doc', '.docx', '.md'}
        for file in files:
            file_ext = '.' + file.filename.split('.')[-1].lower()
            if file_ext not in allowed_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File type {file_ext} not supported. Allowed types: {allowed_types}"
                )
        
        results = service.process_uploaded_files(files)
        
        success_count = sum(1 for r in results if r.get("success", False))
        
        if success_count == 0:
            raise HTTPException(status_code=400, detail="No files were successfully processed")
        
        logger.info(f"✅ Successfully processed {success_count}/{len(files)} files")
        
        return {
            "status": "success",
            "message": f"Processed {success_count}/{len(files)} files",
            "results": results,
            "files_processed": success_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/configure-pipeline")
async def configure_pipeline(config_data: PipelineConfig):
    """Step 2: Configure pipeline with user selections"""
    try:
        logger.info(f"⚙️ Configuring pipeline: {config_data.provider}/{config_data.model}")
        logger.info(f"📝 Chunking: {config_data.chunking_strategy} (size: {config_data.chunk_size}, overlap: {config_data.chunk_overlap})")
        logger.info(f"🧠 Embedder: {config_data.embedder}")
        
        # Validate configuration
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
        
        success = service.configure_pipeline(
            provider=config_data.provider,
            model=config_data.model,
            embedder=config_data.embedder,
            chunking_strategy=config_data.chunking_strategy,
            chunk_size=config_data.chunk_size,
            chunk_overlap=config_data.chunk_overlap
        )
        
        if success:
            logger.info("✅ Pipeline configured successfully")
            return {
                "status": "success",
                "message": "Pipeline configured successfully",
                "configuration": config_data.dict()
            }
        else:
            raise HTTPException(status_code=400, detail="Configuration failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/initialize-pipeline")
async def initialize_pipeline():
    """Step 3: Initialize the configured pipeline"""
    try:
        logger.info("🔄 Initializing pipeline...")
        
        # Check if pipeline is configured
        status = service.get_status()
        if not status["configuration"]["provider"]:
            raise HTTPException(
                status_code=400, 
                detail="Pipeline not configured. Configure pipeline first."
            )
        
        # Check if files are processed
        if status["files_processed"] == 0:
            raise HTTPException(
                status_code=400, 
                detail="No files processed. Upload files first."
            )
        
        success = service.initialize_pipeline()
        
        if success:
            logger.info("🎉 Pipeline initialized and ready for chat")
            return {
                "status": "success",
                "message": "Pipeline initialized and ready for chat"
            }
        else:
            raise HTTPException(status_code=400, detail="Initialization failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """Step 4: Chat with the initialized pipeline"""
    try:
        logger.info(f"💬 Query: {request.question[:100]}...")
        
        # Check if pipeline is ready
        status = service.get_status()
        if not status["ready_for_chat"]:
            raise HTTPException(
                status_code=400, 
                detail="Pipeline not ready. Complete setup first (upload → configure → initialize)."
            )
        
        response, sources = service.query(request.question)
        
        logger.info(f"✅ Response generated with {len(sources)} sources")
        
        return {
            "status": "success",
            "response": response,
            "sources": sources,
            "question": request.question
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get current pipeline status"""
    try:
        status = service.get_status()
        return {
            "status": "success",
            "data": status
        }
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_pipeline():
    """Reset the pipeline to initial state"""
    try:
        logger.info("🔄 Resetting pipeline...")
        success = service.reset()
        
        if success:
            logger.info("✅ Pipeline reset successfully")
            return {
                "status": "success", 
                "message": "Pipeline reset successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Reset failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = service.get_status()
        return {
            "status": "healthy",
            "app_name": config.app_name,
            "version": config.version,
            "pipeline_ready": status["ready_for_chat"],
            "files_processed": status["files_processed"]
        }
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"status": "error", "message": "Endpoint not found"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return {"status": "error", "message": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )