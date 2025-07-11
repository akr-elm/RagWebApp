import logging
import os
from datetime import datetime
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import UnstructuredReader
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, documents_dir):
        self.documents_dir = documents_dir
        self.texts_dir = "data/texts"
        # Create texts directory if it doesn't exist
        os.makedirs(self.texts_dir, exist_ok=True)

    def load_documents(self):
        """Loads .pdf, .txt, .md — uses UnstructuredPDFLoader for PDFs automatically."""
        try:
            reader = SimpleDirectoryReader(
                input_dir=self.documents_dir,
                required_exts=[".pdf", ".txt", ".md"],
                recursive=False,
                file_extractor={".pdf": UnstructuredReader()}
            )
            documents = reader.load_data()
            
            # Add filename to metadata if missing
            for doc in documents:
                print(doc.metadata)
                if "filename" not in doc.metadata:
                    file_path = doc.metadata.get("file_path", "")
                    if file_path:
                        doc.metadata["filename"] = Path(file_path).name
            
            if documents:
                with open("data/texts/extracted_text.txt", "w", encoding="utf-8") as f:
                    f.write(documents[0].text)
           
            # Save extracted texts
            #self._save_extracted_texts(documents)
            
            logger.info(f"📚 Loaded {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    # def _save_extracted_texts(self, documents):
    #     """Save extracted texts to data/texts directory"""
    #     try:
    #         # Clear existing text files
    #         for file_path in Path(self.texts_dir).glob("*_extracted.txt"):
    #             file_path.unlink()
            
    #         saved_files = []
    #         for doc in documents:
    #             filename = doc.metadata.get('filename', 'Unknown')
    #             base_name = Path(filename).stem  # Get filename without extension
    #             output_path = Path(self.texts_dir) / f"{base_name}_extracted.txt"
                
    #             # Save extracted text with metadata
    #             with open(output_path, 'w', encoding='utf-8') as f:
    #                 f.write(f"# Extracted from: {filename}\n")
    #                 f.write(f"# Source path: {doc.metadata.get('file_path', 'Unknown')}\n")
    #                 f.write(f"# Characters: {len(doc.text):,}\n")
    #                 f.write(f"# Words: {len(doc.text.split()):,}\n")
    #                 f.write(f"# Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #                 f.write(f"# Extractor: SimpleDirectoryReader (LlamaIndex)\n")
    #                 f.write("=" * 80 + "\n\n")
    #                 f.write(doc.text)
                
    #             saved_files.append(str(output_path))
            
    #         logger.info(f"💾 Saved {len(saved_files)} extracted text files to {self.texts_dir}")
            
    #         # Log file details
    #         for file_path in saved_files:
    #             file_size = Path(file_path).stat().st_size
    #             logger.info(f"  📄 {Path(file_path).name}: {file_size:,} bytes")
            
    #     except Exception as e:
    #         logger.error(f"❌ Error saving extracted texts: {e}")

    # def get_extracted_texts_info(self):
    #     """Get information about saved extracted texts"""
    #     try:
    #         text_files = list(Path(self.texts_dir).glob("*_extracted.txt"))
            
    #         files_info = []
    #         for file_path in text_files:
    #             stat = file_path.stat()
    #             files_info.append({
    #                 "filename": file_path.name,
    #                 "path": str(file_path),
    #                 "size_bytes": stat.st_size,
    #                 "modified": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    #             })
            
    #         return {
    #             "total_files": len(files_info),
    #             "directory": self.texts_dir,
    #             "files": files_info
    #         }
            
    #     except Exception as e:
    #         logger.error(f"❌ Error getting extracted texts info: {e}")
    #         return {"total_files": 0, "directory": self.texts_dir, "files": []}
        
