from app.core.chunker import Chunker
from app.core.embedder import Embedder
from app.core.vector_store import VectorStoreHandler
from app.core.document_loader import DocumentLoader
from app.core.llm_handler import LLMHandler
from app.config import get_config

import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, provider=None, model_name=None):
        config = get_config()
        self.provider = provider or config.default_llm
        self.model_name = model_name or config.default_model
        self.query_engine = None
        
    def initialize(self, documents_dir=None, chunking_strategy=None, chunk_size=None, chunk_overlap=None):
        """Initialize the complete RAG pipeline"""
        config = get_config()
        
        # Use config defaults if not provided
        documents_dir = documents_dir or config.processed_dir
        chunking_strategy = chunking_strategy or config.chunking_strategy[0]
        chunk_size = chunk_size or config.chunk_size
        chunk_overlap = chunk_overlap or config.chunk_overlap
        try:
            # Initialize components
            logger.info("🔧 Setting up components...")
            document_loader = DocumentLoader(documents_dir)
            embedder = Embedder()
            chunker = Chunker(chunk_size, chunk_overlap, chunking_strategy)
            vector_store = VectorStoreHandler()
            llm_handler = LLMHandler(self.provider, self.model_name)

            # Load documents
            logger.info("📄 Loading documents...")
            documents = document_loader.load_documents()
            if not documents:
                logger.error("❌ No documents found!")
                return False
            logger.info(f"✅ Loaded {len(documents)} documents")

            # Create chunks
            logger.info(f"🔪 Creating chunks using {chunking_strategy} strategy...")
            chunks = chunker.create_chunks(documents)
            
            if not chunks:
                logger.error("❌ Failed to create chunks!")
                return False
            
            logger.info(f"✅ Created {len(chunks)} chunks using {chunking_strategy} strategy")
            
            # Show chunk distribution
            logger.info("📊 Chunk Distribution:")
            doc_chunk_count = {}
            for chunk in chunks:
                filename = chunk.metadata.get('filename', 'Unknown')
                doc_chunk_count[filename] = doc_chunk_count.get(filename, 0) + 1
            
            for filename, count in doc_chunk_count.items():
                logger.info(f"• {filename}: {count} chunks")

            # Create vector index
            logger.info("🔍 Building vector index...")
            index = vector_store.create_index(chunks, embedder.get_embed_model())
            if not index:
                logger.error("❌ Failed to create vector index!")
                return False
            logger.info("✅ Vector index created")

            # Create query engine
            logger.info("⚙️ Creating query engine...")
            self.query_engine = index.as_query_engine(
                llm=llm_handler.get_llm(),
                similarity_top_k=3,
                response_mode="compact"
            )
            
            logger.info("🎉 RAG pipeline ready!")
            return True
                
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False

    def query(self, question):
        """Query the RAG pipeline with source attribution"""
        if not self.query_engine:
            raise ValueError("Pipeline not initialized")
        
        try:
            response = self.query_engine.query(question)
            
            # Debug: Check response structure
            logger.debug(f"Debug - Response type: {type(response)}")
            logger.debug(f"Debug - Response attributes: {dir(response)}")
            
            # Get source information
            sources = []
            source_details = []
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                logger.debug(f"Debug - Found {len(response.source_nodes)} source nodes")
                
                for i, node in enumerate(response.source_nodes):
                    logger.debug(f"Debug - Node {i} metadata: {node.metadata}")
                    
                    filename = node.metadata.get('filename', 'Unknown')
                    sources.append(filename)
                    source_details.append({
                        'filename': filename,
                        'text_preview': node.text[:100] + "..." if len(node.text) > 100 else node.text
                    })
            else:
                logger.debug("Debug - No source_nodes found in response")
            
            # Create response with sources
            response_text = str(response)
            if sources:
                unique_sources = list(set(sources))  # Remove duplicates
                source_text = "\n\n**Sources:** " + ", ".join(unique_sources)
                response_text += source_text
            else:
                response_text += "\n\n**Sources:** No sources identified"
            
            return response_text, source_details
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise