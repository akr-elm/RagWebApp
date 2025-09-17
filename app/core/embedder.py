from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name="LaBSE"):
        self.model_name = model_name
        
        # Set cache directories for offline usage according to HF docs
        self.cache_dir = Path("/app/models_cache")
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir)
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(self.cache_dir)
        
        # For offline mode (according to HF docs)
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Load available models from cache directory scan (skip broken JSON file)
        self.available_models = self.load_available_models()
        logger.info(f"Available models type: {type(self.available_models)}")
        logger.info(f"Available models content: {self.available_models}")
        
        # Get working models
        working_models = []
        if isinstance(self.available_models, dict):
            working_models = [k for k, v in self.available_models.items() if v.get('status') == 'available']
            logger.info(f"Working models from dict: {working_models}")
        else:
            logger.error(f"Expected dict but got {type(self.available_models)}: {self.available_models}")
            # Fallback - if it's somehow a list, assume all are available
            if isinstance(self.available_models, list):
                working_models = self.available_models
                logger.info(f"Using list as working models: {working_models}")
        
        logger.info(f"Final working models: {working_models}")
        
        if model_name not in working_models:
            logger.warning(f"Requested model '{model_name}' not available. Available models: {working_models}")
            if working_models:
                model_name = working_models[0]  # Use first available model
                logger.info(f"Using fallback model: {model_name}")
            else:
                raise RuntimeError("No embedding models are available. Check if models were downloaded during build.")
        
        try:
            # Initialize with cache folder
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                cache_folder=str(self.cache_dir)
            )
            self.model_name = model_name
            logger.info(f"✅ Embedder initialized: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedder with {model_name}: {e}")
            
            # Try fallback to simplest model
            if model_name != "all-MiniLM-L6-v2" and "all-MiniLM-L6-v2" in working_models:
                try:
                    logger.info("Trying fallback to all-MiniLM-L6-v2...")
                    self.embed_model = HuggingFaceEmbedding(
                        model_name="all-MiniLM-L6-v2",
                        cache_folder=str(self.cache_dir)
                    )
                    self.model_name = "all-MiniLM-L6-v2"
                    logger.info("✅ Fallback embedder initialized: all-MiniLM-L6-v2")
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    raise RuntimeError(f"Could not initialize any embedding model. Original error: {e}")
            else:
                raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def get_embed_model(self):
        return self.embed_model
    
    def load_available_models(self):
        """Load available models - always use cache directory scan"""
        logger.info("Using cache directory scan to detect available models")
        scan_result = self.scan_cache_directory()
        logger.info(f"Scan result: {scan_result}")
        return scan_result
    
    def scan_cache_directory(self):
        """Scan cache directory for models"""
        available = {}
        expected_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2", 
            "LaBSE"
        ]
        
        logger.info(f"Scanning cache directory: {self.cache_dir}")
        if not self.cache_dir.exists():
            logger.warning(f"Cache directory does not exist: {self.cache_dir}")
            return available
            
        logger.info(f"Cache dir contents: {list(self.cache_dir.iterdir())}")
        
        # Show the actual structure we found
        for item in self.cache_dir.iterdir():
            if item.is_dir() and "sentence-transformers" in item.name:
                logger.info(f"Found HF model directory: {item.name}")
        
        for model_name in expected_models:
            try:
                # Check the HuggingFace cache format
                hf_model_dir = self.cache_dir / f"models--sentence-transformers--{model_name}"
                
                logger.info(f"Checking for {model_name} at: {hf_model_dir}")
                
                if hf_model_dir.exists():
                    logger.info(f"Directory exists for {model_name}")
                    available[model_name] = {
                        "status": "available", 
                        "cache_path": str(hf_model_dir)
                    }
                    logger.info(f"✅ Found model {model_name} at {hf_model_dir}")
                else:
                    logger.warning(f"❌ Model directory does not exist: {hf_model_dir}")
                    available[model_name] = {"status": "not_found"}
                    
            except Exception as e:
                logger.error(f"Error checking model {model_name}: {e}")
                available[model_name] = {"status": "error", "error": str(e)}
        
        logger.info(f"Final scan results: {available}")
        return available
    
    def get_available_cached_models(self):
        """Get list of models that are actually cached and working"""
        if isinstance(self.available_models, dict):
            cached_models = [k for k, v in self.available_models.items() if v.get('status') == 'available']
        else:
            cached_models = list(self.available_models) if self.available_models else []
        logger.info(f"Available cached models: {cached_models}")
        return cached_models
    
    @staticmethod
    def get_available_models():
        """Get list of embedding models that should be available"""
        return [
            "all-MiniLM-L6-v2",        # Fast, good performance
            "LaBSE",                   # Language-agnostic BERT
        ]