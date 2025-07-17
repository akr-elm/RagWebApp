from llama_index.llms.ollama import Ollama
import logging

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, provider='Ollama', model_name='llama3.2:1b'):
        self.provider = provider 
        self.model_name = model_name 
        self.llm = None  # Initialize the llm attribute
        
        # Initialize the LLM during construction
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        try:

            self.llm = self._create_ollama_llm()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise
    
    def _create_ollama_llm(self):
        """Create Ollama LLM instance"""
        return Ollama(
            model=self.model_name,
            temperature=0,
            request_timeout=100
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