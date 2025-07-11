import logging
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
import chromadb
from app.config import get_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreHandler:
    """Handles vector store operations"""
    def __init__(self, persist_dir=None):
        config = get_config()
        self.persist_dir = persist_dir or config.vector_store_path
        self.vector_store_type = config.vector_store_type
        self.collection_name = config.vector_store_collection_name or "default_collection"
        try:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            logger.info(f"✅ Initialized vector store: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Error initializing vector store: {e}")
            raise


    def create_index(self, chunks, embed_model):
        try:
            index = VectorStoreIndex(
                nodes=chunks, 
                vector_store=self.vector_store,
                embed_model=embed_model
            )
            logger.info(f"✅ Created vector index with {len(chunks)} chunks")
            return index
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            return None
