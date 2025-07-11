from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os
from pathlib import Path

@dataclass
class Config:
    """
    Configuration class for the application.
    """
    # General settings
    app_name: str = "FAST-APP"
    version: str = "1.0.0"
    
    # Raw document storage
    raw_dir: str = './data/raw'

    # Processed document storage
    processed_dir: str = './data/documents'

    #LLM settings
    llm_provider: List[str] = field(default_factory=lambda: ['groq', 'ollama'])
    default_llm: str = 'groq'
    default_model: str = 'gemma2-9b-it'
    ollama_models: List[str] = field(default_factory=lambda: ['llama3.2:1b', 'llama3.2:3b', 'mistral:7b','tinyllama:latest'])
    temperature: float = 0.1
    max_tokens: int = 1024
    request_timeout: float = 600.0

    # Chunking settings
    chunking_strategy: List[str] = field(default_factory=lambda: [
        "fixed", 
        "recursive", 
        "langchain_recursive",  # New
        "token",               # New
        "semantic"
    ])
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # Document loader settings
    document_types: List[str] = field(default_factory=lambda: ['txt', 'pdf'])
    max_file_size_mb: int = 50

    # Vector store settings
    vector_store_type: str = 'chroma'
    vector_store_path: str = './vector_store'
    vector_store_collection_name: str = 'test_collection'
    
    # API Keys
    groq_api_key: Optional[str] = None

    # Available LLM providers and models
    available_providers: Dict[str, List[str]] = field(default_factory=lambda: {
    "groq": ["llama3-8b-8192", "llama3-70b-8192", "gemma2-9b-it", "mixtral-8x7b-32768"],
    "ollama": ["llama3.2:1b", "llama3.2:3b", "mistral:7b","tinyllama:latest"],
    })
    
     # Available embedders
    available_embedders: List[str] = field(default_factory=lambda: [
            "all-MiniLM-L6-v2",        # Fast, good performance (multilingual)
            "all-mpnet-base-v2",       # Best quality, slower (multilingual)
            "paraphrase-multilingual-MiniLM-L12-v2",  # Good for French
            "distiluse-base-multilingual-cased",      # Multilingual, good balance
            "LaBSE",                  # Language-agnostic BERT (excellent for French)
    ])

    def __post_init__(self):
        """Initialize default values and load from environment"""
        # Load from environment variables
        self.groq_api_key = os.getenv("GROQ_API_KEY", self.groq_api_key)
        self.default_llm = os.getenv("DEFAULT_LLM", self.default_llm)
        self.default_model = os.getenv("DEFAULT_MODEL", self.default_model)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", str(self.chunk_size)))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(self.chunk_overlap)))
        
        # Create directories
        Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self, provider) -> Optional[str]:
        """Get API key for the specified provider"""
        if provider.lower() == 'groq':
            return os.getenv("GROQ_API_KEY")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        if self.default_llm == 'groq' and not self.groq_api_key:
            errors.append("GROQ_API_KEY is required when using Groq")
        
        if self.chunk_overlap >= self.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")
        
        if errors:
            for error in errors:
                print(f"Config Error: {error}")
            return False
        return True

# Global config instance
config = Config()

# Convenience function
def get_config() -> Config:
    """Get the global configuration instance"""
    return config