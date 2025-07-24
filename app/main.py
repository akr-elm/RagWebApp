from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import config
from app.api.routes import documents, pipeline, chat, system
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_app() -> FastAPI:
    """Application factory"""
    # Validate config on startup
    if not config.validate():
        raise Exception("Invalid configuration")
    
    app = FastAPI(
        title=config.app_name,
        description="Clean RAG Pipeline API",
        version=config.version
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include all routes
    app.include_router(documents.router, prefix="/api")
    app.include_router(pipeline.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(system.router, prefix="/api")
    
    # Serve the HTML interface
    @app.get("/test")
    async def serve_test_interface():
        return FileResponse('testing_interface.html')
    
    @app.get("/")
    async def root():
        return {
            "message": f"🚀 {config.app_name} is running!",
            "version": config.version,
            "docs": "/docs",
            "test_interface": "/test",  # Add this
            "config_valid": config.validate(),
            "api_endpoints": {
                "upload": "/api/documents/upload",
                "configure": "/api/pipeline/configure",
                "initialize": "/api/pipeline/initialize",
                "chat": "/api/chat/",
                "status": "/api/system/status",
                "options": "/api/system/options",
                "health": "/api/system/health"
            }
        }
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)