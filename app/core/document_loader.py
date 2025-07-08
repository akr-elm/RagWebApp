import os
import logging
from pathlib import Path
import shutil
from llama_index.core import SimpleDirectoryReader, Document
from app.config import get_config

# Setup logging
logger = logging.getLogger(__name__)

class FileProcessor:
    """Simple file processor for PDF and TXT files"""
    
    def __init__(self, raw_dir="data/raw", processed_dir="data/documents"):
        config = get_config()
        self.raw_dir = config.raw_dir or raw_dir
        self.processed_dir = config.processed_dir or processed_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def process_files(self, uploaded_files):
        """Process multiple uploaded files - main method"""
        results = []
        
        # Clear old files
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            try:
                # Save file
                raw_path = os.path.join(self.raw_dir, uploaded_file.filename)
                with open(raw_path, "wb") as f:
                    f.write(uploaded_file.file.read())
                
                # Process based on extension
                content = ""
                if uploaded_file.filename.lower().endswith('.pdf'):
                    content = self._extract_pdf(raw_path)
                elif uploaded_file.filename.lower().endswith('.txt'):
                    content = self._extract_txt(raw_path)
                else:
                    logger.warning(f"Unsupported file type: {uploaded_file.filename}")
                    continue
                
                # Save as processed .txt file
                if content:
                    base_name = Path(uploaded_file.filename).stem
                    processed_path = os.path.join(self.processed_dir, f"{base_name}.txt")
                    with open(processed_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    results.append({
                        'original': uploaded_file.filename,
                        'processed': f"{base_name}.txt",
                        'success': True,
                        'size': len(content)
                    })
                    logger.info(f"✅ Processed: {uploaded_file.filename}")
                else:
                    results.append({
                        'original': uploaded_file.filename,
                        'success': False
                    })
                    logger.error(f"❌ Failed: {uploaded_file.filename}")
                    
            except Exception as e:
                results.append({
                    'original': uploaded_file.filename,
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"Error processing {uploaded_file.filename}: {e}")
        
        return results
    
    def _extract_pdf(self, file_path):
        """Extract text from PDF with multiple fallback methods"""
        
        # Method 1: Try PyPDF2 first (fastest)
        text = self._try_pypdf2(file_path)
        if text and not text.startswith("["):  # Check if extraction was successful
            return text
        
        # Method 2: Try OCR as last resort (slowest but works on scanned PDFs)
        text = self._try_ocr(file_path)
        if text and not text.startswith("["):
            return text
        
        # If all methods fail
        return "[PDF content could not be extracted with any available method]"

    def _try_pypdf2(self, file_path):
        """Try extracting PDF text using PyPDF2"""
        try:
            import PyPDF2
            text = ""
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    logger.warning(f"PDF is encrypted: {file_path}")
                    return "[PDF is encrypted - cannot extract text]"
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                        continue
            
            if text.strip():
                logger.info(f"✅ PyPDF2 successfully extracted text from: {file_path}")
                return text
            else:
                logger.warning(f"PyPDF2 extracted no text from: {file_path}")
                return "[PyPDF2 extracted no readable text]"
                
        except ImportError:
            logger.warning("PyPDF2 not installed")
            return "[PyPDF2 not installed]"
        except Exception as e:
            logger.error(f"PyPDF2 error for {file_path}: {e}")
            return f"[PyPDF2 error: {e}]"

    def _try_ocr(self, file_path):
        """Try extracting PDF text using OCR (for scanned PDFs)"""
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            logger.info(f"Attempting OCR extraction for: {file_path}")
            
            # Convert PDF to images
            try:
                images = convert_from_path(file_path, dpi=200, first_page=1, last_page=5)  # Limit to first 5 pages
            except Exception as e:
                logger.error(f"Failed to convert PDF to images: {e}")
                return f"[PDF to image conversion failed: {e}]"
            
            text = ""
            for i, img in enumerate(images):
                try:
                    # Extract text from image using OCR
                    page_text = pytesseract.image_to_string(img, lang='eng')
                    if page_text.strip():
                        text += f"\n--- Page {i + 1} (OCR) ---\n"
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"OCR failed on page {i + 1}: {e}")
                    continue
            
            if text.strip():
                logger.info(f"✅ OCR successfully extracted text from: {file_path}")
                return text
            else:
                return "[OCR extracted no readable text]"
                
        except ImportError as e:
            missing_deps = []
            if "pdf2image" in str(e):
                missing_deps.append("pdf2image")
            if "pytesseract" in str(e):
                missing_deps.append("pytesseract")
            
            logger.warning(f"OCR dependencies not installed: {missing_deps}")
            return f"[OCR dependencies not installed: {', '.join(missing_deps)}]"
        except Exception as e:
            logger.error(f"OCR error for {file_path}: {e}")
            return f"[OCR error: {e}]"
            
    def _extract_txt(self, file_path):
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"[Text file error: {e}]"
    
    def get_processed_files(self):
        """Get list of processed files"""
        if not os.path.exists(self.processed_dir):
            return []
        return list(Path(self.processed_dir).glob("*.txt"))
    
    def clear_files(self):
        """Clear all files"""
        try:
            if os.path.exists(self.raw_dir):
                shutil.rmtree(self.raw_dir)
            if os.path.exists(self.processed_dir):
                shutil.rmtree(self.processed_dir)
            os.makedirs(self.raw_dir, exist_ok=True)
            os.makedirs(self.processed_dir, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error clearing files: {e}")
            return False

import os
from pathlib import Path
from typing import List, Optional
import logging

from llama_index.core import SimpleDirectoryReader
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, documents_dir: str):
        self.documents_dir = documents_dir
    
    def load_documents(self):
        """Load documents using LlamaIndex SimpleDirectoryReader"""
        try:
            # Use LlamaIndex's built-in document loader
            reader = SimpleDirectoryReader(
                input_dir=self.documents_dir,
                required_exts=[".txt", ".pdf", ".md"],
                recursive=False
            )
            
            documents = reader.load_data()
            
            # Add filename to metadata if not present
            for doc in documents:
                if 'filename' not in doc.metadata:
                    # Extract filename from file_path if available
                    file_path = doc.metadata.get('file_path', '')
                    if file_path:
                        doc.metadata['filename'] = Path(file_path).name
            
            logger.info(f"📚 Loaded {len(documents)} documents using SimpleDirectoryReader")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    