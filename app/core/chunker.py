from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser

import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Chunker:
    """Handles document chunking"""
    def __init__(self, chunk_size=800, chunk_overlap=100, chunk_strategy='fixed'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy.lower()

    def create_chunks(self, documents):
        """Create chunks based on the selected strategy"""
        if not documents:
            return []
        
        if self.chunk_strategy == 'semantic':
            return self.create_semantic_chunks(documents)
        else:
            return self.create_fixed_chunks(documents)
    
    def create_fixed_chunks(self, documents):
        """Create fixed-size chunks"""
        try:
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.get_nodes_from_documents(documents)
            logger.info(f"✅ Created {len(chunks)} fixed chunks")
            return chunks
        except Exception as e:
            logger.error(f"❌ Error creating fixed chunks: {e}")
            return []
    
    def create_semantic_chunks(self, documents):
        """Create semantic chunks using SemanticSplitterNodeParser"""
        if not documents:
            return []
        try:
            # Initialize the semantic splitter
            splitter = SemanticSplitterNodeParser(
                embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                buffer_size=1,  # Number of sentences to group together
                breakpoint_percentile_threshold=95,  # Threshold for semantic breaks
                include_metadata=True,
                include_prev_next_rel=True
            )
            
            # Create semantic chunks
            chunks = splitter.get_nodes_from_documents(documents)
            
            # Debug: Check chunk metadata preservation
            for i, chunk in enumerate(chunks[:3]):
                logger.info(f"Semantic chunk {i} metadata: {chunk.metadata}")
            
            logger.info(f"✅ Created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error creating semantic chunks: {e}")
            return []   
    