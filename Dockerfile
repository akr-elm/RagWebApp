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
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper cache directories
RUN mkdir -p data/raw data/documents data/texts vector_store logs \
    /app/models_cache /root/.cache/huggingface /root/.cache/torch

# Set environment variables for model caching BEFORE downloading
ENV SENTENCE_TRANSFORMERS_HOME=/app/models_cache
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV TORCH_HOME=/app/models_cache
ENV HF_HOME=/app/models_cache

# Pre-download embedding models one by one to avoid memory issues
RUN python -c "import os; os.makedirs('/app/models_cache', exist_ok=True)"

# Download models individually with proper error handling and verification
RUN python -c "\
import os; \
from sentence_transformers import SentenceTransformer; \
print('Downloading all-MiniLM-L6-v2...'); \
print('✅ all-MiniLM-L6-v2 downloaded'); \
print('Model saved at:', model1._modules['0'].auto_model.config._name_or_path); \
del model1"


RUN python -c "\
import os; \
from sentence_transformers import SentenceTransformer; \
print('Downloading LaBSE...'); \
model2 = SentenceTransformer('LaBSE'); \
print('✅ LaBSE downloaded'); \
del model2"

# Verify what was actually downloaded and where
RUN echo "=== CHECKING DOWNLOADED MODELS ===" && \
    find /app/models_cache -type f -name "*.bin" -o -name "*.safetensors" -o -name "config.json" | head -20 && \
    echo "=== CHECKING SENTENCE TRANSFORMERS DIRECTORY ===" && \
    find /app/models_cache -type d -name "*sentence*" && \
    echo "=== CHECKING MODEL DIRECTORIES ===" && \
    ls -la /app/models_cache/ && \
    echo "=== END MODEL CHECK ==="

# Copy the application code AFTER downloading models
COPY . .

# Set proper permissions
RUN chmod -R 755 data/ vector_store/ logs/ models_cache/

# Create a model verification script
RUN python -c "\
import os; \
import json; \
from pathlib import Path; \
from sentence_transformers import SentenceTransformer; \
\
# Test loading each model to verify they work \
models_info = {}; \
test_models = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'LaBSE']; \
\
for model_name in test_models: \
    try: \
        print(f'Testing {model_name}...'); \
        model = SentenceTransformer(model_name); \
        # Test encoding \
        test_embedding = model.encode('test text'); \
        models_info[model_name] = { \
            'status': 'available', \
            'embedding_dim': len(test_embedding), \
            'cache_path': str(Path(model.cache_folder) / model_name) \
        }; \
        print(f'✅ {model_name}: embedding_dim={len(test_embedding)}'); \
        del model; \
    except Exception as e: \
        models_info[model_name] = {'status': 'failed', 'error': str(e)}; \
        print(f'❌ {model_name}: {e}'); \
\
# Save model info \
with open('/app/available_models.json', 'w') as f: \
    json.dump(models_info, f, indent=2); \
\
print('Available models saved to /app/available_models.json')"

# Show the results
RUN cat /app/available_models.json

# Set environment variables for offline mode
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0

# Command to run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]