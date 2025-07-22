from functools import lru_cache
from app.config import config, get_config
from app.services.ingestion_service import IngestionService

@lru_cache()
def get_settings():
    """Return your dataclass config (alias for compatibility)"""
    return get_config()

@lru_cache()
def get_config_instance():
    """Direct access to config"""
    return get_config()

@lru_cache()
def get_ingestion_service():
    return IngestionService()