from app.core.chunker import Chunker
from app.core.embedder import Embedder
from app.core.vector_store import VectorStoreHandler
from app.core.document_loader import DocumentLoader
from app.core.llm_handler import LLMHandler
from llama_index.core import PromptTemplate
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(self, provider='Ollama', model_name='gemma2:2b', embedder_model=None):
        self.provider = provider 
        self.model_name = model_name
        self.embedder_model = embedder_model or "LaBSE"  # Default embedder
        self.query_engine = None
        
    def initialize(self, documents_dir='data/raw', chunking_strategy=None, chunk_size=None, chunk_overlap=None, embedder_model=None):
        """Initialize the complete RAG pipeline"""
    
        # Provide proper defaults instead of passing None
        documents_dir = documents_dir or 'data/raw'
        chunking_strategy = chunking_strategy or 'langchain_recursive'  # Default strategy
        chunk_size = chunk_size or 512  # Default chunk size
        chunk_overlap = chunk_overlap or 50  # Default overlap
        embedder_model = embedder_model or self.embedder_model
        
        try:
            # Initialize components
            logger.info("🔧 Setting up components...")
            logger.info(f"📋 Configuration: strategy={chunking_strategy}, size={chunk_size}, overlap={chunk_overlap}")

            document_loader = DocumentLoader(documents_dir)
            
            # Initialize embedder with specified model
            logger.info(f"🧠 Initializing embedder: {embedder_model}")
            embedder = Embedder(model_name=embedder_model)
            
            # Initialize chunker with validated parameters
            chunker = Chunker(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                strategy=chunking_strategy
            )
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
            
            qa_prompt_tmpl = (
                "Les informations contextuelles sont ci-dessous.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Instructions :\n"
                "1. Lisez et analysez attentivement les informations contextuelles ci-dessus\n"
                "2. Répondez à la question en utilisant UNIQUEMENT les informations fournies dans le contexte\n"
                "3. Répondez dans la MÊME LANGUE que celle de la question\n"
                "4. Si le contexte contient des informations pertinentes, fournissez une réponse complète\n"
                "5. Si vous ne trouvez pas de réponse directe, fournissez des informations connexes issues du contexte qui pourraient être utiles\n"
                "6. Si le contexte ne contient aucune information pertinente, indiquez-le clairement\n"
                "7. Citez toujours les parties du contexte que vous avez utilisées dans votre réponse\n"
                "\n"
                "Question : {query_str}\n"
                "\n"
                "Réponse (dans la même langue que la question) :"
            )


            # Create prompt template
            qa_prompt = PromptTemplate(qa_prompt_tmpl)

            self.query_engine = index.as_query_engine(
                llm=llm_handler.get_llm(),
                similarity_top_k=top_k,
                response_mode="tree_summarize",
                text_qa_template=qa_prompt
            )
            
            logger.info(f"🎉 RAG pipeline ready! Using top_k={top_k} for retrieval")
            return True
                
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False

    def query(self, question):
        """Query the RAG pipeline with source attribution and chunk printing"""
        if not self.query_engine:
            raise ValueError("Pipeline not initialized")
        
        try:
            response = self.query_engine.query(question)
            
            # Print retrieved chunks
            logger.info(f"🔍 QUESTION: {question}")
            logger.info("=" * 80)
            
            # Get source information with chunk details
            sources = []
            source_details = []
            seen_files = set()
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                logger.info(f"📚 RETRIEVED {len(response.source_nodes)} CHUNKS:")
                logger.info("=" * 80)
                
                for i, node in enumerate(response.source_nodes):
                    filename = node.metadata.get('filename', 'Unknown')
                    chunk_id = node.metadata.get('chunk_id', 'N/A')
                    chunk_size = len(node.text)
                    score = getattr(node, 'score', 'N/A')
                    
                    # Print chunk details
                    logger.info(f"CHUNK #{i+1}:")
                    logger.info(f"  📄 File: {filename}")
                    logger.info(f"  🆔 Chunk ID: {chunk_id}")
                    logger.info(f"  📏 Size: {chunk_size} chars")
                    logger.info(f"  📊 Score: {score}")
                    logger.info(f"  📝 Content:")
                    logger.info(f"     {node.text[:200]}..." if len(node.text) > 200 else f"     {node.text}")
                    logger.info("-" * 60)
                    
                    # Add to sources if new file
                    if filename not in seen_files:
                        seen_files.add(filename)
                        source_details.append({
                            'filename': filename,
                            'text_preview': node.text[:100] + "..." if len(node.text) > 100 else node.text,
                            'chunk_id': chunk_id,
                            'score': score,
                            'size': chunk_size
                        })
            else:
                logger.info("❌ No chunks retrieved!")
            
            logger.info("=" * 80)
            logger.info(f"💬 RESPONSE: {str(response)}")
            logger.info("=" * 80)
            
            return str(response), source_details
            
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