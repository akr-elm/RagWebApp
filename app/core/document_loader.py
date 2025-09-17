import logging
import os
import time
from datetime import datetime
from llama_index.core import SimpleDirectoryReader, Document
from pathlib import Path
import PyPDF2

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, documents_dir):
        self.documents_dir = documents_dir
        self.texts_dir = "data/texts"
        # Create texts directory if it doesn't exist
        os.makedirs(self.texts_dir, exist_ok=True)

    def load_documents(self):
        """Loads .pdf, .txt, .md, .docx — uses smart fallback extraction for PDFs and DOCX."""
        try:
            documents = []
            
            # Get all files - added .docx support
            files = []
            for ext in [".pdf", ".txt", ".md", ".docx"]:
                pattern = f"*{ext}"
                files.extend(Path(self.documents_dir).glob(pattern))
            
            if not files:
                logger.warning(f"No documents found in {self.documents_dir}")
                return []
            
            logger.info(f"Found {len(files)} files to process")
            
            for file_path in files:
                try:
                    if file_path.suffix.lower() == ".pdf":
                        # Use smart fallback extraction for PDF files
                        text, extractor = self._extract_pdf_with_fallbacks(file_path)
                    elif file_path.suffix.lower() == ".docx":
                        # Use DOCX extraction with fallbacks
                        text, extractor = self._extract_docx_with_fallbacks(file_path)
                    else:
                        # Use simple text reading for .txt and .md
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        extractor = "text_reader"
                    
                    if text.strip():  # Only create document if we got text
                        # Create LlamaIndex Document
                        doc = Document(
                            text=text,
                            metadata={
                                "filename": file_path.name,
                                "file_path": str(file_path),
                                "extractor": extractor,
                                "file_size": file_path.stat().st_size,
                                "created_at": datetime.now().isoformat(),
                                "character_count": len(text)
                            }
                        )
                        documents.append(doc)
                        logger.info(f"✅ Processed {file_path.name}: {len(text):,} characters using {extractor}")
                    else:
                        logger.warning(f"⚠️ No text extracted from {file_path.name}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path.name}: {e}")
                    continue
            
            # Add filename to metadata and print metadata for debugging
            for doc in documents:
                print(f"Document metadata: {doc.metadata}")
            
            # Save extracted texts for review
            if documents:
                self._save_extracted_texts(documents)
           
            logger.info(f"📚 Successfully loaded {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []
    
    def _extract_docx_with_fallbacks(self, docx_path):
        """
        Extract DOCX text with multiple fallbacks:
        1. python-docx (fastest, best for simple documents)
        2. LlamaIndex SimpleDirectoryReader (good for complex documents)
        3. Unstructured (handles complex layouts, tables)
        """
        
        docx_path = Path(docx_path)
        if not docx_path.exists():
            logger.error(f"File not found: {docx_path}")
            return "", "error"
        
        logger.info(f"📄 Processing DOCX: {docx_path.name}")
        results = []
        
        # Method 1: python-docx (fastest, good for basic documents)
        try:
            logger.info("🔄 Trying python-docx...")
            start_time = time.time()
            
            from docx import Document as DocxDocument
            
            doc = DocxDocument(str(docx_path))
            text = ""
            
            # Extract paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                text += "\n--- TABLE ---\n"
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    text += " | ".join(row_text) + "\n"
                text += "--- END TABLE ---\n\n"
            
            if len(text.strip()) > 50:  # Reasonable amount of text
                duration = time.time() - start_time
                logger.info(f"✅ python-docx succeeded: {len(text):,} chars in {duration:.2f}s")
                return text, "python-docx"
            else:
                logger.warning(f"⚠️ python-docx: Only {len(text)} characters extracted")
                results.append(("python-docx", text, len(text)))
                
        except ImportError:
            logger.warning("❌ python-docx not installed. Install with: pip install python-docx")
        except Exception as e:
            logger.warning(f"❌ python-docx failed: {e}")
        
        # Method 2: LlamaIndex SimpleDirectoryReader (built-in support)
        try:
            logger.info("🔄 Trying LlamaIndex SimpleDirectoryReader...")
            start_time = time.time()
            
            from llama_index.core import SimpleDirectoryReader
            
            loader = SimpleDirectoryReader(
                input_files=[str(docx_path)],
                filename_as_id=True
            )
            
            documents = loader.load_data()
            
            if documents and len(documents[0].text.strip()) > 50:
                text = "\n\n".join([doc.text for doc in documents])
                duration = time.time() - start_time
                logger.info(f"✅ LlamaIndex succeeded: {len(text):,} chars in {duration:.2f}s")
                return text, "LlamaIndex-SimpleDirectoryReader"
            else:
                text = "\n\n".join([doc.text for doc in documents]) if documents else ""
                logger.warning(f"⚠️ LlamaIndex: Only {len(text)} characters extracted")
                results.append(("LlamaIndex-SimpleDirectoryReader", text, len(text)))
                
        except Exception as e:
            logger.warning(f"❌ LlamaIndex SimpleDirectoryReader failed: {e}")
        
        # Method 3: Unstructured (handles complex layouts)
        try:
            logger.info("🔄 Trying Unstructured...")
            start_time = time.time()
            
            from llama_index.readers.file import UnstructuredReader
            
            unstructured_reader = UnstructuredReader()
            documents = unstructured_reader.load_data(str(docx_path))
            
            if documents and len(documents[0].text.strip()) > 50:
                text = "\n\n".join([doc.text for doc in documents])
                duration = time.time() - start_time
                logger.info(f"✅ Unstructured succeeded: {len(text):,} chars in {duration:.2f}s")
                return text, "Unstructured"
            else:
                text = "\n\n".join([doc.text for doc in documents]) if documents else ""
                logger.warning(f"⚠️ Unstructured: Only {len(text)} characters extracted")
                results.append(("Unstructured", text, len(text)))
                
        except Exception as e:
            logger.warning(f"❌ Unstructured failed: {e}")
        
        # Method 4: mammoth (for complex Word documents with formatting)
        try:
            logger.info("🔄 Trying mammoth...")
            start_time = time.time()
            
            import mammoth
            
            with open(str(docx_path), "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text = result.value
                
                if result.messages:
                    logger.info(f"Mammoth messages: {result.messages}")
            
            if len(text.strip()) > 50:
                duration = time.time() - start_time
                logger.info(f"✅ mammoth succeeded: {len(text):,} chars in {duration:.2f}s")
                return text, "mammoth"
            else:
                logger.warning(f"⚠️ mammoth: Only {len(text)} characters extracted")
                results.append(("mammoth", text, len(text)))
                
        except ImportError:
            logger.warning("❌ mammoth not installed. Install with: pip install mammoth")
        except Exception as e:
            logger.warning(f"❌ mammoth failed: {e}")
        
        # If all methods failed, return the best result we got
        if results:
            best_result = max(results, key=lambda x: x[2])  # Sort by character count
            method, text, char_count = best_result
            logger.warning(f"⚠️ All primary methods had issues. Using best result from {method}: {char_count} chars")
            return text, f"{method}-fallback"
        
        logger.error("❌ All DOCX extraction methods failed!")
        return "", "failed"

    def _extract_pdf_with_fallbacks(self, pdf_path, use_ocr_fallback=True):
        """
        Extract PDF text with multiple fallbacks:
        1. PyPDF2 (fastest, good for tables)
        2. PyMuPDF (fast, reliable)
        3. pdfplumber (good for structured documents)
        4. Unstructured fast (no OCR) - optional
        5. Unstructured with OCR (slowest, last resort) - optional
        """
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"File not found: {pdf_path}")
            return "", "error"
        
        logger.info(f"📄 Processing PDF: {pdf_path.name}")
        results = []
        
        # Method 1: PyPDF2 (fast, good for tables)
        try:
            logger.info("🔄 Trying PyPDF2...")
            start_time = time.time()
            
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n--- PAGE {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        logger.warning(f"Could not extract page {page_num + 1}: {e}")
                        continue
            
            if len(text.strip()) > 100:  # Reasonable amount of text
                duration = time.time() - start_time
                logger.info(f"✅ PyPDF2 succeeded: {len(text):,} chars in {duration:.2f}s")
                return text, "PyPDF2"
            else:
                logger.warning(f"⚠️ PyPDF2: Only {len(text)} characters extracted")
                results.append(("PyPDF2", text, len(text)))
                
        except Exception as e:
            logger.warning(f"❌ PyPDF2 failed: {e}")
      
        # Method 3: pdfplumber (good for structured documents)
        try:
            logger.info("🔄 Trying pdfplumber...")
            start_time = time.time()
            
            import pdfplumber
            text = ""
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- PAGE {page_num + 1} ---\n"
                        text += page_text
            
            if len(text.strip()) > 100:
                duration = time.time() - start_time
                logger.info(f"✅ pdfplumber succeeded: {len(text):,} chars in {duration:.2f}s")
                return text, "pdfplumber"
            else:
                logger.warning(f"⚠️ pdfplumber: Only {len(text)} characters extracted")
                results.append(("pdfplumber", text, len(text)))
                
        except Exception as e:
            logger.warning(f"❌ pdfplumber failed: {e}")
        
        # Method 4: LlamaIndex SimpleDirectoryReader with Unstructured (fast, no OCR)
        if use_ocr_fallback:
            try:
                logger.info("🔄 Trying LlamaIndex with Unstructured (fast, no OCR)...")
                start_time = time.time()
                
                from llama_index.core import SimpleDirectoryReader
                from llama_index.readers.file import UnstructuredReader
                
                # Create UnstructuredReader with fast strategy
                unstructured_reader = UnstructuredReader()
                
                # Use SimpleDirectoryReader with the unstructured reader
                loader = SimpleDirectoryReader(
                    input_files=[str(pdf_path)],
                    file_extractor={".pdf": unstructured_reader},
                    filename_as_id=True
                )
                
                documents = loader.load_data()
                
                if documents and len(documents[0].text.strip()) > 100:
                    text = "\n\n".join([doc.text for doc in documents])
                    duration = time.time() - start_time
                    logger.info(f"✅ LlamaIndex + Unstructured (fast) succeeded: {len(text):,} chars in {duration:.2f}s")
                    return text, "LlamaIndex-Unstructured-fast"
                else:
                    text = "\n\n".join([doc.text for doc in documents]) if documents else ""
                    logger.warning(f"⚠️ LlamaIndex + Unstructured (fast): Only {len(text)} characters extracted")
                    results.append(("LlamaIndex-Unstructured-fast", text, len(text)))
                    
            except Exception as e:
                logger.warning(f"❌ LlamaIndex + Unstructured (fast) failed: {e}")
            
            # Method 5: LlamaIndex SimpleDirectoryReader with Unstructured OCR (very slow, last resort)
            try:
                logger.warning("🔄 Trying LlamaIndex with Unstructured OCR (this may take several minutes)...")
                logger.warning("⏳ Please wait... OCR processing can be very slow")
                start_time = time.time()
                
                from llama_index.core import SimpleDirectoryReader
                from llama_index.readers.file import UnstructuredReader
                
                # Create UnstructuredReader with hi_res strategy for OCR
                unstructured_reader = UnstructuredReader(
                    # Configure for OCR processing
                    partition_kwargs={
                        "strategy": "hi_res",
                        "infer_table_structure": True,
                        "extract_images": False,
                        "languages": ["eng", "fra"]
                    }
                )
                
                # Use SimpleDirectoryReader with the OCR-enabled unstructured reader
                loader = SimpleDirectoryReader(
                    input_files=[str(pdf_path)],
                    file_extractor={".pdf": unstructured_reader},
                    filename_as_id=True
                )
                
                documents = loader.load_data()
                
                if documents:
                    text = "\n\n".join([doc.text for doc in documents])
                    duration = time.time() - start_time
                    logger.info(f"✅ LlamaIndex + Unstructured OCR succeeded: {len(text):,} chars in {duration:.2f}s")
                    return text, "LlamaIndex-Unstructured-OCR"
                else:
                    logger.error("❌ No documents returned from Unstructured OCR")
                    
            except Exception as e:
                logger.error(f"❌ LlamaIndex + Unstructured OCR failed: {e}")
                results.append(("LlamaIndex-Unstructured-OCR", "", 0))    
        # If all methods failed, return the best result we got
        if results:
            best_result = max(results, key=lambda x: x[2])  # Sort by character count
            method, text, char_count = best_result
            logger.warning(f"⚠️ All primary methods had issues. Using best result from {method}: {char_count} chars")
            return text, f"{method}-fallback"
        
        logger.error("❌ All extraction methods failed!")
        return "", "failed"
    
    def _save_extracted_texts(self, documents):
        """Save extracted texts to files for review"""
        try:
            # Save all documents to one file
            with open(os.path.join(self.texts_dir, "extracted_text.txt"), "w", encoding="utf-8") as f:
                for i, doc in enumerate(documents):
                    f.write(f"\n{'='*80}\n")
                    f.write(f"DOCUMENT {i+1}: {doc.metadata.get('filename', 'Unknown')}\n")
                    f.write(f"EXTRACTOR: {doc.metadata.get('extractor', 'Unknown')}\n")
                    f.write(f"SIZE: {len(doc.text):,} characters\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(doc.text)
                    f.write(f"\n\n")
            
            # Save first document separately (for compatibility)
            if documents:
                with open(os.path.join(self.texts_dir, "first_document.txt"), "w", encoding="utf-8") as f:
                    f.write(documents[0].text)
                    
            logger.info(f"💾 Saved extracted texts to {self.texts_dir}/")
            
        except Exception as e:
            logger.error(f"Error saving extracted texts: {e}")
    
    def check_if_pdf_needs_ocr(self, pdf_path):
        """Quick check to see if PDF might be image-based (needs OCR)"""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            
            total_chars = 0
            image_count = 0
            
            for page_num, page in enumerate(doc):
                # Get text
                text = page.get_text()
                total_chars += len(text.strip())
                
                # Check for images
                image_list = page.get_images()
                image_count += len(image_list)
                
                if page_num >= 2:  # Check first 3 pages only
                    break
            
            doc.close()
            
            logger.info(f"📊 PDF Analysis for {pdf_path.name}:")
            logger.info(f"  Characters in first 3 pages: {total_chars}")
            logger.info(f"  Images in first 3 pages: {image_count}")
            
            if total_chars < 50 and image_count > 0:
                logger.info("🔍 This PDF likely needs OCR (appears to be scanned/image-based)")
                return True
            elif total_chars > 200:
                logger.info("📝 This PDF has extractable text (OCR not needed)")
                return False
            else:
                logger.info("❓ Unclear if OCR is needed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Could not analyze PDF: {e}")
            return None