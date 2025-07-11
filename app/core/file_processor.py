import os
import shutil
import logging
from pathlib import Path
from app.config import get_config

logger = logging.getLogger(__name__)

class FileProcessor:
    """
    Saves uploaded files directly — no manual PDF extraction needed!
    """

    def __init__(self, raw_dir="data/raw", processed_dir="data/documents"):
        config = get_config()
        self.raw_dir = config.raw_dir or raw_dir
        self.processed_dir = config.processed_dir or processed_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def process_files(self, uploaded_files):
        """Save files to processed_dir"""
        results = []

        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            try:
                ext = uploaded_file.filename.lower().split(".")[-1]

                if ext not in ["pdf", "txt", "md"]:
                    logger.warning(f"Unsupported file type: {uploaded_file.filename}")
                    continue

                path = os.path.join(self.processed_dir, uploaded_file.filename)
                with open(path, "wb") as f:
                    f.write(uploaded_file.file.read())

                results.append({
                    "original": uploaded_file.filename,
                    "processed": uploaded_file.filename,
                    "success": True,
                })

                logger.info(f"✅ Saved: {uploaded_file.filename}")

            except Exception as e:
                results.append({
                    "original": uploaded_file.filename,
                    "success": False,
                    "error": str(e)
                })
                logger.error(f"Error saving {uploaded_file.filename}: {e}")

        return results

    def clear_files(self):
        """Clear raw & processed files"""
        try:
            shutil.rmtree(self.raw_dir, ignore_errors=True)
            shutil.rmtree(self.processed_dir, ignore_errors=True)
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.processed_dir, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error clearing files: {e}")
            return False

    def get_processed_files(self):
        """List saved files"""
        if not os.path.exists(self.processed_dir):
            return []
        return list(Path(self.processed_dir).glob("*"))
