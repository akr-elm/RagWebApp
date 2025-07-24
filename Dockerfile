# Use Python 3.11 as base image
FROM python:3.11-slim                                                                  

# Set working directory
WORKDIR /app

# Install system dependencies for OCR + PDF
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    curl \
    software-properties-common \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/documents data/texts vector_store embedders_cache logs

# Expose port
EXPOSE 8000

# Environment config
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Pre-download embedding models to cache them
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
SentenceTransformer('all-mpnet-base-v2'); \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
SentenceTransformer('LaBSE'); \
print('✅ Embedding models downloaded successfully')"

# Set proper permissions
RUN chmod -R 755 data/ vector_store/ embedders_cache/ logs/

# Command to run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]