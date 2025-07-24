from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PipelineConfig(BaseModel):
    provider: str
    model: str
    embedder: str
    chunking_strategy: str = "langchain_recursive"
    chunk_size: int = Field(default=800, ge=100, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

class ChatResponse(BaseModel):
    status: str
    response: str
    sources: List[Dict[str, Any]]
    question: str

class StatusResponse(BaseModel):
    ready_for_chat: bool
    files_processed: int
    configuration: Optional[Dict[str, Any]] = None

class SuccessResponse(BaseModel):
    status: str = "success"
    message: str
    data: Optional[Dict[str, Any]] = None