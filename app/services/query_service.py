# app/services/query_service.py

import logging

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def query(self, question: str):
        logger.info(f"🔍 Querying: {question}")
        answer, sources = self.pipeline.query(question)
        return {
            "answer": answer,
            "sources": sources
        }
