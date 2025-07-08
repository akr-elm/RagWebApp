from llama_index.llms.groq import Groq
from llama_index.llms.ollama import Ollama
from app.config import get_config
import logging

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, provider=None, model_name=None):
        config = get_config()
        self.provider = provider or config.default_llm
        self.model_name = model_name or config.default_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.request_timeout = config.request_timeout
        self.llm = None  # Initialize the llm attribute
        
        # Initialize the LLM during construction
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        try:
            if self.provider.lower() == "groq":
                self.llm = self._create_groq_llm()
            elif self.provider.lower() == "ollama":
                self.llm = self._create_ollama_llm()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                
            logger.info(f"✅ LLM initialized: {self.provider} - {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise
    
    def _create_groq_llm(self):
        """Create Groq LLM instance"""
        config = get_config()
        api_key = config.get_api_key("groq")
        
        if not api_key:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY environment variable.")
        
        return Groq(
            model=self.model_name,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout
        )
    
    def _create_ollama_llm(self):
        """Create Ollama LLM instance"""
        return Ollama(
            model=self.model_name,
            temperature=self.temperature,
            request_timeout=self.request_timeout
        )
    
    def get_llm(self):
        """Get the initialized LLM instance"""
        if self.llm is None:
            self._initialize_llm()
        return self.llm
    
    def query(self, prompt: str):
        """Query the LLM with a prompt"""
        try:
            if self.llm is None:
                raise ValueError("LLM not initialized")
            
            response = self.llm.complete(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        try:
            return self.llm is not None
        except Exception:
            return False