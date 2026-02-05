# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --force && npm rebuild esbuild
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend with Unstructured dependencies
FROM python:3.11-slim AS backend

# Install system dependencies for Unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    pandoc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/app ./app

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static

# Create data directory
RUN mkdir -p /data/knowledgevault

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
