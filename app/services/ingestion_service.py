from app.core.document_loader import FileProcessor
from app.core.rag_pipeline import RAGPipeline
from app.core.embedder import Embedder
from app.config import get_config
import logging

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.processor = FileProcessor()
        self.pipeline = None
        self.config = get_config()
        self.current_setup = {
            "provider": None,
            "model": None,
            "embedder": None,
            "initialized": False
        }

    def get_available_options(self):
        """Get available providers, models, and embedders"""
        return {
            "providers": self.config.available_providers,
            "embedders": self.config.available_embedders,
            "chunking_strategies": ["fixed", "semantic", "recursive"]
        }

    def process_files(self, uploaded_files):
        """Step 1: Process uploaded files"""
        results = self.processor.process_files(uploaded_files)
        logger.info(f"📁 Processed {len(uploaded_files)} files")
        return results

    def configure_pipeline(self, provider, model, embedder, chunking_strategy="fixed", chunk_size=800, chunk_overlap=100):
        """Step 2: Configure pipeline with user selections"""
        try:
            # Validate selections
            if provider not in self.config.available_providers:
                raise ValueError(f"Invalid provider: {provider}")
            
            if model not in self.config.available_providers[provider]:
                raise ValueError(f"Invalid model for {provider}: {model}")
            
            if embedder not in self.config.available_embedders:
                raise ValueError(f"Invalid embedder: {embedder}")

            # Create new pipeline with selections
            self.pipeline = RAGPipeline(provider=provider, model_name=model)
            
            # Store current setup
            self.current_setup = {
                "provider": provider,
                "model": model,
                "embedder": embedder,
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "initialized": False
            }
            
            logger.info(f"🔧 Pipeline configured: {provider}/{model} with {embedder} embedder")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration failed: {e}")
            return False

    def initialize_pipeline(self):
        """Step 3: Initialize the configured pipeline"""
        if not self.pipeline:
            raise ValueError("Pipeline not configured. Call configure_pipeline() first.")
        
        try:
            # Initialize with user selections - Remove embedder_type parameter
            success = self.pipeline.initialize(
                documents_dir=self.processor.processed_dir,
                chunking_strategy=self.current_setup["chunking_strategy"],
                chunk_size=self.current_setup["chunk_size"],
                chunk_overlap=self.current_setup["chunk_overlap"]
                # Remove this line: embedder_type=self.current_setup["embedder"]
            )
            
            if success:
                self.current_setup["initialized"] = True
                logger.info("🎉 Pipeline initialized successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False

    def get_pipeline(self):
        """Get the initialized pipeline for querying"""
        if not self.current_setup["initialized"]:
            raise ValueError("Pipeline not initialized")
        return self.pipeline

    def get_status(self):
        """Get current pipeline status"""
        try:
            # Try to get processed files count safely
            processed_files = 0
            if hasattr(self.processor, 'get_processed_files'):
                try:
                    processed_files = len(self.processor.get_processed_files())
                except:
                    processed_files = 0
            
            return {
                "files_processed": processed_files,
                "configuration": self.current_setup,
                "ready_for_chat": self.current_setup["initialized"]
            }
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                "files_processed": 0,
                "configuration": self.current_setup,
                "ready_for_chat": False
            }

    def reset(self):
        """Reset the service to initial state"""
        self.pipeline = None
        self.current_setup = {
            "provider": None,
            "model": None,
            "embedder": None,
            "initialized": False
        }
        logger.info("🔄 Service reset")