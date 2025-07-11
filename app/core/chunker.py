from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
from app.config import get_config
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Chunker:
    """Handles document chunking with multiple strategies"""
    
    def __init__(self, chunk_size=800, chunk_overlap=100, chunk_strategy='fixed'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy.lower()

    def create_chunks(self, documents):
        """Create chunks from documents using the specified strategy"""
        try:
            if self.chunk_strategy == "semantic":
                # Semantic chunking works with ALL documents at once
                chunks = self._semantic_chunk(documents)
            else:
                # Other strategies work document by document
                all_chunks = []
                
                for doc in documents:
                    if self.chunk_strategy == "recursive":
                        chunks = self._recursive_chunk(doc)
                    elif self.chunk_strategy == "token":
                        chunks = self._token_chunk_langchain(doc)
                    elif self.chunk_strategy == "langchain_recursive":
                        chunks = self._langchain_recursive_chunk(doc)
                    else:  # fixed (default)
                        chunks = self._fixed_chunk(doc)
                    
                    all_chunks.extend(chunks)
                
                chunks = all_chunks
                
            logger.info(f"✅ Created {len(chunks)} total chunks from {len(documents)} documents using {self.chunk_strategy} strategy")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Chunking failed: {e}")
            return []

    def _fixed_chunk(self, document):
        """Create fixed-size chunks from a single document"""
        text = document.text
        chunks = []
        
        logger.debug(f"Fixed chunking document: {document.metadata.get('filename', 'Unknown')} ({len(text)} chars)")
        
        # Always chunk if document is larger than chunk_size
        if len(text) > self.chunk_size:
            # Normal chunking for larger documents
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                chunks.append(Document(
                    text=chunk_text,
                    metadata={**document.metadata, 'chunk_id': chunk_id}
                ))
                
                start = end - self.chunk_overlap
                chunk_id += 1
                
                # Safety break
                if chunk_id > 1000:
                    logger.warning(f"Breaking chunking loop at {chunk_id} chunks")
                    break
                    
        elif len(text) > 200:
            # Small document handling - split into 2 chunks
            mid_point = len(text) // 2
            
            # Try to find a good break point (sentence end)
            for i in range(mid_point, min(mid_point + 200, len(text))):
                if text[i] in '.!?\n':
                    mid_point = i + 1
                    break
            
            chunks.append(Document(
                text=text[:mid_point],
                metadata={**document.metadata, 'chunk_id': 0}
            ))
            chunks.append(Document(
                text=text[mid_point:],
                metadata={**document.metadata, 'chunk_id': 1}
            ))
        else:
            # Very small document - keep as single chunk
            chunks.append(Document(
                text=text,
                metadata={**document.metadata, 'chunk_id': 0}
            ))
        
        logger.debug(f"Fixed chunking created {len(chunks)} chunks")
        return chunks

    def _recursive_chunk(self, document):
        """Create recursive chunks using LlamaIndex HierarchicalNodeParser"""
        try:
            logger.debug(f"Recursive chunking document: {document.metadata.get('filename', 'Unknown')} ({len(document.text)} chars)")

            # Define inner splitter for the final fallback level
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                include_metadata=True
            )

            # Hierarchical parser tries splitting by structure first
            parser = HierarchicalNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                sentence_splitter=splitter
            )

            chunks = parser.get_nodes_from_documents([document])

            if len(chunks) == 1 and len(document.text) > self.chunk_size:
                logger.debug(f"HierarchicalNodeParser failed to split large doc, falling back to LangChain")
                return self._langchain_recursive_chunk(document)

            logger.debug(f"Hierarchical chunking created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"❌ Error in recursive chunking: {e}")
            logger.info("Falling back to LangChain recursive chunking")
            return self._langchain_recursive_chunk(document)

    def _langchain_recursive_chunk(self, document):
        """Use LangChain's RecursiveCharacterTextSplitter - most reliable"""
        try:
            logger.debug(f"LangChain recursive chunking: {document.metadata.get('filename', 'Unknown')} ({len(document.text)} chars)")
            
            # LangChain's RecursiveCharacterTextSplitter with smart separators
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",  # Paragraph break
                ]
            )
            
            # Split the text
            chunks_text = text_splitter.split_text(document.text)
            
            # Convert back to LlamaIndex Document format
            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(Document(
                        text=chunk_text.strip(),
                        metadata={**document.metadata, 'chunk_id': i}
                    ))
            
            logger.debug(f"LangChain recursive chunking created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error in LangChain recursive chunking: {e}")
            return self._fixed_chunk(document)

    def _token_chunk_langchain(self, document):
        """Use LangChain's TokenTextSplitter for precise token-based chunking"""
        try:
            logger.debug(f"LangChain token chunking: {document.metadata.get('filename', 'Unknown')} ({len(document.text)} chars)")
            
            # Token-based splitter
            text_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            
            # Split the text
            chunks_text = text_splitter.split_text(document.text)
            
            # Convert back to LlamaIndex Document format
            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                if chunk_text.strip():
                    chunks.append(Document(
                        text=chunk_text.strip(),
                        metadata={**document.metadata, 'chunk_id': i}
                    ))
            
            logger.debug(f"LangChain token chunking created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error in LangChain token chunking: {e}")
            return self._fixed_chunk(document)

    def _character_chunk_langchain(self, document):
        """Use LangChain's CharacterTextSplitter for simple character-based chunking"""
        try:
            logger.debug(f"LangChain character chunking: {document.metadata.get('filename', 'Unknown')} ({len(document.text)} chars)")
            
            # Character-based splitter
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=" "
            )
            
            # Split the text
            chunks_text = text_splitter.split_text(document.text)
            
            # Convert back to LlamaIndex Document format
            chunks = []
            for i, chunk_text in enumerate(chunks_text):
                if chunk_text.strip():
                    chunks.append(Document(
                        text=chunk_text.strip(),
                        metadata={**document.metadata, 'chunk_id': i}
                    ))
            
            logger.debug(f"LangChain character chunking created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error in LangChain character chunking: {e}")
            return self._fixed_chunk(document)

    def _semantic_chunk(self, documents):
        """Create semantic chunks using SemanticSplitterNodeParser - works on ALL documents"""
        if not documents:
            return []
        try:
            logger.debug(f"Semantic chunking {len(documents)} documents")
            
            # Initialize the semantic splitter
            splitter = SemanticSplitterNodeParser(
                embed_model=HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                buffer_size=1,
                breakpoint_percentile_threshold=90,
                include_metadata=True,
                include_prev_next_rel=False  # Disable to avoid issues
            )
            
            # Create semantic chunks from ALL documents at once
            chunks = splitter.get_nodes_from_documents(documents)
            
            # Debug: Check chunk metadata preservation
            for i, chunk in enumerate(chunks[:3]):
                logger.debug(f"Semantic chunk {i} metadata: {chunk.metadata}")
            
            logger.debug(f"Semantic chunking created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error creating semantic chunks: {e}")
            logger.info("Falling back to LangChain recursive chunking")
            # Fallback to LangChain recursive chunking
            all_chunks = []
            for doc in documents:
                chunks = self._langchain_recursive_chunk(doc)
                all_chunks.extend(chunks)
            return all_chunks

    def _manual_chunk(self, document):
        """Manual chunking as final fallback"""
        text = document.text
        chunks = []
        
        logger.debug(f"Manual chunking document: {document.metadata.get('filename', 'Unknown')} ({len(text)} chars)")
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunks.append(Document(
                text=chunk_text,
                metadata={**document.metadata, 'chunk_id': chunk_id}
            ))
            
            start = end - self.chunk_overlap
            chunk_id += 1
            
            if chunk_id > 1000:  # Safety break
                logger.warning(f"Breaking manual chunking loop at {chunk_id} chunks")
                break
        
        logger.debug(f"Manual chunking created {len(chunks)} chunks")
        return chunks

        return descriptions.get(strategy, "Unknown strategy")