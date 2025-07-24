from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole
from app.config import get_config
from dotenv import load_dotenv
import os
import logging

load_dotenv()  # Load environment variables from .env file
logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, provider='ollama', model_name='gemma2:2b'):
        config = get_config()
        self.provider = provider.lower()
        self.model_name = model_name
        self.llm = None
        
        # Initialize the LLM during construction
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        try:
            if self.provider == "groq":
                self.llm = self._create_groq_llm()
            elif self.provider == "ollama":
                self.llm = self._create_ollama_llm()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
                
            logger.info(f"✅ LLM initialized: {self.provider}/{self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            raise
    
    def _create_groq_llm(self):
        """Create Groq LLM instance"""        
        # Get API key from config
        api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            raise ValueError("Groq API key not found. Please set groq_api_key in config.")
        
        return Groq(
            model=self.model_name,
            api_key=api_key,
            temperature=0.1,
            max_tokens=1024,
            request_timeout=100
        )
    
    def _create_ollama_llm(self):
        """Create Ollama LLM instance"""
        return Ollama(
            model=self.model_name,
            temperature=0,
            request_timeout=100,
            base_url='http://localhost:11434'
        )
    
    def get_llm(self):
        """Get the initialized LLM instance"""
        if self.llm is None:
            self._initialize_llm()
        return self.llm
    
    def query(self, prompt: str, context: str = None, use_chat_format: bool = True):
        """Query the LLM with a prompt"""
        try:
            if self.llm is None:
                raise ValueError("LLM not initialized")
            
            if use_chat_format:
                # Use chat format
                messages = []
                
                # Add context if provided
                if context:
                    user_content = f"Context: {context}\n\nQuestion: {prompt}"
                else:
                    user_content = prompt
                
                messages.append(ChatMessage(role=MessageRole.USER, content=user_content))
                
                response = self.llm.chat(messages)
                return response.message.content
            else:
                # Use simple completion
                if context:
                    full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
                else:
                    full_prompt = prompt
                
                response = self.llm.complete(full_prompt)
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