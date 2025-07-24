from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embed_model = HuggingFaceEmbedding(model_name=f"sentence-transformers/{model_name}")
        logger.info(f"✅ Embedder initialized: {model_name}")
    
    def get_embed_model(self):
        return self.embed_model
    
    @staticmethod
    def get_available_models():
        """Get list of available embedding models"""
        return [
            "all-MiniLM-L6-v2",        # Fast, good performance (multilingual)
            "all-mpnet-base-v2",       # Best quality, slower (multilingual)
            "paraphrase-multilingual-MiniLM-L12-v2",  # Good for French
            "distiluse-base-multilingual-cased",      # Multilingual, good balance
            "LaBSE",                   # Language-agnostic BERT (excellent for French)
        ]