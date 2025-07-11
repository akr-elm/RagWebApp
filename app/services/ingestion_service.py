from app.core.file_processor import FileProcessor
from app.core.document_loader import DocumentLoader
from app.core.rag_pipeline import RAGPipeline
from app.config import get_config
import logging

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.config = get_config()
        self.processor = FileProcessor()  # Updated to use new FileProcessor
        self.document_loader = DocumentLoader(self.processor.raw_dir)  # Updated
        self.pipeline = None
        self.current_setup = {
            "provider": None,
            "model": None,
            "embedder": None,
            "chunking_strategy": None,
            "chunk_size": None,
            "chunk_overlap": None,
            "initialized": False
        }

    def process_uploaded_files(self, uploaded_files):
        """Step 1: Process uploaded files"""
        try:
            # Clear previous files and process new ones
            results = self.processor.process_files(uploaded_files)
            
            # Update document loader with new processed directory
            self.document_loader = DocumentLoader(self.processor.raw_dir)
            
            # Reset pipeline when new files are uploaded
            self.pipeline = None
            self.current_setup["initialized"] = False
            
            success_count = sum(1 for r in results if r["success"])
            logger.info(f"✅ Processed {success_count}/{len(uploaded_files)} files")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ File processing failed: {e}")
            return []

    def configure_pipeline(self, provider, model, embedder, chunking_strategy="langchain_recursive", chunk_size=800, chunk_overlap=100):
        """Step 2: Configure pipeline with user selections"""
        try:
            # Validate selections
            if provider not in self.config.available_providers:
                raise ValueError(f"Invalid provider: {provider}")
            
            if model not in self.config.available_providers[provider]:
                raise ValueError(f"Invalid model for {provider}: {model}")
            
            if embedder not in self.config.available_embedders:
                raise ValueError(f"Invalid embedder: {embedder}")
            
            if chunking_strategy not in self.config.chunking_strategy:
                raise ValueError(f"Invalid chunking strategy: {chunking_strategy}")

            # Create new pipeline with ALL selections
            self.pipeline = RAGPipeline(
                provider=provider, 
                model_name=model,
                embedder_model=embedder
            )
            
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
            
            logger.info(f"🔧 Pipeline configured: {provider}/{model} with {embedder} embedder, {chunking_strategy} chunking")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration failed: {e}")
            return False

    def initialize_pipeline(self):
        """Step 3: Initialize the configured pipeline"""
        if not self.pipeline:
            raise ValueError("Pipeline not configured. Call configure_pipeline() first.")
        
        try:
            # Check if we have processed files
            processed_files = self.processor.get_processed_files()
            if not processed_files:
                raise ValueError("No files processed. Upload files first.")
            
            # Initialize with user selections
            success = self.pipeline.initialize(
                documents_dir=self.processor.processed_dir,
                chunking_strategy=self.current_setup["chunking_strategy"],
                chunk_size=self.current_setup["chunk_size"],
                chunk_overlap=self.current_setup["chunk_overlap"],
                embedder_model=self.current_setup["embedder"]
            )
            
            if success:
                self.current_setup["initialized"] = True
                logger.info("🎉 Pipeline initialized successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False

    def query(self, question):
        """Step 4: Query the pipeline"""
        if not self.pipeline or not self.current_setup["initialized"]:
            raise ValueError("Pipeline not initialized. Complete setup first.")
        
        return self.pipeline.query(question)

    def get_status(self):
        """Get current service status"""
        try:
            processed_files = self.processor.get_processed_files()
            
            return {
                "files_processed": len(processed_files),
                "files_list": [f.name for f in processed_files],
                "configuration": self.current_setup,
                "ready_for_chat": self.current_setup["initialized"]
            }
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {
                "files_processed": 0,
                "files_list": [],
                "configuration": self.current_setup,
                "ready_for_chat": False
            }

    def reset(self):
        """Reset the service to initial state"""
        try:
            # Clear files
            self.processor.clear_files()
            
            # Reset pipeline
            self.pipeline = None
            self.current_setup = {
                "provider": None,
                "model": None,
                "embedder": None,
                "chunking_strategy": None,
                "chunk_size": None,
                "chunk_overlap": None,
                "initialized": False
            }
            
            logger.info("🔄 Service reset complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Reset failed: {e}")
            return False

    def get_available_options(self):
        """Get all available configuration options"""
        return {
            "providers": self.config.available_providers,
            "embedders": self.config.available_embedders,
            "chunking_strategies": self.config.chunking_strategy,
            "chunk_size_range": {"min": 100, "max": 2000, "default": 800},
            "chunk_overlap_range": {"min": 0, "max": 500, "default": 100}
        }