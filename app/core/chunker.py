from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser, HierarchicalNodeParser
from llama_index.core.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Chunker:
    """Simple document chunker with multiple strategies"""
    
    def __init__(self, chunk_size=512, chunk_overlap=50, strategy='recursive'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy.lower()
        
    def create_chunks(self, documents):
        """Chunk documents using the specified strategy"""
        if not documents:
            return []
            
        try:
            logger.info(f"Chunking {len(documents)} documents with {self.strategy} strategy")
            
            if self.strategy == 'semantic':
                return self._semantic_chunk(documents)
            elif self.strategy == 'hierarchical':
                return self._hierarchical_chunk(documents)
            elif self.strategy == 'langchain':
                return self._langchain_chunk(documents)
            else:  # default: recursive
                return self._recursive_chunk(documents)
                
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            return self._fallback_chunk(documents)
    
    def _recursive_chunk(self, documents):
        """LlamaIndex sentence-aware chunking"""
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.get_nodes_from_documents(documents)
        self._add_metadata(chunks, 'recursive')
        return chunks
    
    def _langchain_chunk(self, documents):
        """LangChain recursive text splitter"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            texts = splitter.split_text(doc.text)
            for i, text in enumerate(texts):
                if text.strip():
                    chunk = Document(text=text.strip(), metadata=doc.metadata.copy())
                    chunks.append(chunk)
        
        self._add_metadata(chunks, 'langchain')
        return chunks
    
    def _semantic_chunk(self, documents):
        """Semantic chunking based on content similarity"""
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=90
        )
        chunks = splitter.get_nodes_from_documents(documents)
        self._add_metadata(chunks, 'semantic')
        return chunks
    
    def _hierarchical_chunk(self, documents):
        """Hierarchical chunking with small and large chunks"""
        leaf_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        parent_parser = SentenceSplitter(
            chunk_size=self.chunk_size * 3,
            chunk_overlap=self.chunk_overlap * 2
        )
        
        hierarchical_parser = HierarchicalNodeParser.from_defaults(
            node_parser_ids=["leaf", "parent"],
            node_parser_map={
                "leaf": leaf_parser,
                "parent": parent_parser
            }
        )
        
        chunks = hierarchical_parser.get_nodes_from_documents(documents)
        self._add_metadata(chunks, 'hierarchical')
        return chunks
    
    def _fallback_chunk(self, documents):
        """Simple fallback chunking"""
        chunks = []
        for doc in documents:
            text = doc.text
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                
                chunk = Document(
                    text=chunk_text,
                    metadata={**doc.metadata, 'chunk_id': chunk_id}
                )
                chunks.append(chunk)
                
                start = end - self.chunk_overlap
                chunk_id += 1
                
                if start >= len(text):
                    break
        
        self._add_metadata(chunks, 'fallback')
        return chunks
    
    def _add_metadata(self, chunks, strategy):
        """Add chunking metadata to chunks"""
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                chunk.metadata = {}
                
            chunk.metadata.update({
                'chunk_id': i,
                'chunk_strategy': strategy,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'chunk_length': len(chunk.text),
                'created_at': datetime.now().isoformat()
            })
    
    def get_strategy_info(self):
        """Get information about available strategies"""
        strategies = {
            'recursive': 'Sentence-aware chunking (default)',
            'langchain': 'Multi-separator recursive chunking',
            'semantic': 'Content similarity-based chunking',
            'hierarchical': 'Multi-level chunking (small + large chunks)'
        }
        return strategies