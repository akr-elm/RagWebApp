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
    
    def __init__(self, provider=None, model_name=None, embedder_model=None):
        config = get_config()
        self.provider = provider or config.default_llm
        self.model_name = model_name or config.default_model
        self.embedder_model = embedder_model or "all-MiniLM-L6-v2"  # Default embedder
        self.query_engine = None
        
    def initialize(self, documents_dir=None, chunking_strategy=None, chunk_size=None, chunk_overlap=None, embedder_model=None):
        """Initialize the complete RAG pipeline"""
        config = get_config()
        
        # Use config defaults if not provided
        documents_dir = documents_dir or config.processed_dir
        chunking_strategy = chunking_strategy or config.chunking_strategy[0]
        chunk_size = chunk_size or config.chunk_size
        chunk_overlap = chunk_overlap or config.chunk_overlap
        embedder_model = embedder_model or self.embedder_model
        
        try:
            # Initialize components
            logger.info("🔧 Setting up components...")
    
            document_loader = DocumentLoader(documents_dir)
            
            # Initialize embedder with specified model
            logger.info(f"🧠 Initializing embedder: {embedder_model}")
            embedder = Embedder(model_name=embedder_model)
            
            # Initialize chunker with better parameters for multiple documents
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

            # Debug: Show document sizes before chunking
            logger.info("📋 Document Analysis:")
            total_chars = 0
            for i, doc in enumerate(documents):
                doc_length = len(doc.text)
                total_chars += doc_length
                filename = doc.metadata.get('filename', f'Document_{i}')
                logger.info(f"• {filename}: {doc_length:,} characters")
            
            logger.info(f"📊 Total content: {total_chars:,} characters")
            logger.info(f"⚙️ Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

            # Create chunks with enhanced logging
            logger.info(f"🔪 Creating chunks using {chunking_strategy} strategy...")
            chunks = chunker.create_chunks(documents)
            
            if not chunks:
                logger.error("❌ Failed to create chunks!")
                return False
            
            logger.info(f"✅ Created {len(chunks)} chunks using {chunking_strategy} strategy")
            
            # Enhanced chunk distribution analysis
            logger.info("📊 Detailed Chunk Distribution:")
            doc_chunk_count = {}
            doc_chunk_sizes = {}
            
            for i, chunk in enumerate(chunks):
                filename = chunk.metadata.get('filename', 'Unknown')
                chunk_size_actual = len(chunk.text)
                
                # Count chunks per document
                doc_chunk_count[filename] = doc_chunk_count.get(filename, 0) + 1
                
                # Track chunk sizes per document
                if filename not in doc_chunk_sizes:
                    doc_chunk_sizes[filename] = []
                doc_chunk_sizes[filename].append(chunk_size_actual)
                
                # Log individual chunks if there are few
                if len(chunks) <= 10:
                    logger.info(f"  Chunk {i+1}: {filename} ({chunk_size_actual:,} chars)")
            
            # Summary per document
            for filename, count in doc_chunk_count.items():
                avg_size = sum(doc_chunk_sizes[filename]) / len(doc_chunk_sizes[filename])
                min_size = min(doc_chunk_sizes[filename])
                max_size = max(doc_chunk_sizes[filename])
                logger.info(f"• {filename}: {count} chunks (avg: {avg_size:.0f}, min: {min_size}, max: {max_size} chars)")

            # Warning if chunking seems ineffective
            if len(chunks) == len(documents):
                logger.warning("⚠️ Each document became exactly 1 chunk. Consider:")
                logger.warning("   - Reducing chunk_size (current: {})".format(chunk_size))
                logger.warning("   - Check if documents are very small")
                logger.warning("   - Verify chunking strategy is working")

            # Create vector index
            logger.info(f"🔍 Building vector index with {embedder_model}...")
            index = vector_store.create_index(chunks, embedder.get_embed_model())
            if not index:
                logger.error("❌ Failed to create vector index!")
                return False
            logger.info("✅ Vector index created")

            # Create query engine with adjusted parameters for multiple documents
            logger.info("⚙️ Creating query engine...")
            
            # Adjust similarity_top_k based on number of chunks and documents
            top_k = min(5, max(2, len(chunks) // 2))  # Dynamic top_k
            
            self.query_engine = index.as_query_engine(
                llm=llm_handler.get_llm(),
                similarity_top_k=top_k,
                response_mode="compact"
            )
            
            logger.info(f"🎉 RAG pipeline ready! Using top_k={top_k} for retrieval")
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
            
            # Get source information with deduplication
            sources = []
            source_details = []
            seen_files = set()  # Track files we've already seen
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                logger.debug(f"Debug - Found {len(response.source_nodes)} source nodes")
                
                for i, node in enumerate(response.source_nodes):
                    logger.debug(f"Debug - Node {i} metadata: {node.metadata}")
                    
                    filename = node.metadata.get('filename', 'Unknown')
                    
                    # Only add if we haven't seen this file before
                    if filename not in seen_files:
                        seen_files.add(filename)
                        source_details.append({
                            'filename': filename,
                            'text_preview': node.text[:100] + "..." if len(node.text) > 100 else node.text
                        })
            else:
                logger.debug("Debug - No source_nodes found in response")
            
            # Return ONLY the response text WITHOUT appending sources
            response_text = str(response)
            
            # Don't append sources to response_text - let the frontend handle it
            return response_text, source_details
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise
    def get_configuration(self):
        """Get current pipeline configuration"""
        return {
            "provider": self.provider,
            "model": self.model_name,
            "embedder": self.embedder_model,
            "initialized": self.query_engine is not None
        }