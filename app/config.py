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
    

    #LLM settings
    llm_provider: List[str] = field(default_factory=lambda: [ 'ollama','groq'])
    default_llm: str = 'ollama'
    default_model: str = 'gemma2:2b'
    ollama_models: List[str] = field(default_factory=lambda: ['llama3.2:1b','gemma2:2b', 'qwen2.5:1.5b', 'llama3.2:3b', 'mistral:7b','tinyllama:latest'])

    # Chunking settings
    chunking_strategy: List[str] = field(default_factory=lambda: [
        "fixed", 
        "recursive", 
        "langchain_recursive",  # New
        "hierarchical",
        "semantic"
    ])

    # Document loader settings
    document_types: List[str] = field(default_factory=lambda: ['txt', 'pdf'])
    max_file_size_mb: int = 50


    # Available LLM providers and models
    available_providers: Dict[str, List[str]] = field(default_factory=lambda: {
    "ollama": ["gemma2:2b", "qwen2.5:1.5b", "llama3.2:3b", "mistral:7b","tinyllama:latest"],
    "groq": ["gemma2-9b-it", "qwen2.5:7b"]
    })

     # Available embedders
    available_embedders: List[str] = field(default_factory=lambda: [
            "all-MiniLM-L6-v2",        # Fast, good performance (multilingual)
            "all-mpnet-base-v2",       # Best quality, slower (multilingual)
            "paraphrase-multilingual-MiniLM-L12-v2",  # Good for French
            "distiluse-base-multilingual-cased",      # Multilingual, good balance
            "LaBSE",                  # Language-agnostic BERT (excellent for French)
    ])

    def validate(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check if required fields are present
            if not self.app_name:
                raise ValueError("app_name is required")
            
            if not self.version:
                raise ValueError("version is required")
            
            # Check if lists are not empty
            if not self.llm_provider:
                raise ValueError("llm_provider cannot be empty")
            
            if not self.available_providers:
                raise ValueError("available_providers cannot be empty")
            
            if not self.available_embedders:
                raise ValueError("available_embedders cannot be empty")
            
            if not self.chunking_strategy:
                raise ValueError("chunking_strategy cannot be empty")
            
            # Check if default values are valid
            if self.default_llm not in self.llm_provider:
                raise ValueError(f"default_llm '{self.default_llm}' not in llm_provider")
            
            if self.default_llm not in self.available_providers:
                raise ValueError(f"default_llm '{self.default_llm}' not in available_providers")
            
            if self.default_model not in self.available_providers.get(self.default_llm, []):
                raise ValueError(f"default_model '{self.default_model}' not available for provider '{self.default_llm}'")
            
            # Check file size limit
            if self.max_file_size_mb <= 0:
                raise ValueError("max_file_size_mb must be positive")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Global config instance
config = Config()

# Convenience function
def get_config() -> Config:
    """Get the global configuration instance"""
    return config