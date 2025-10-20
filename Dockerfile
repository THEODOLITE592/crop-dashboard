# 1. Base image
FROM python:3.11-slim

# 2. Set workdir
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy backend requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy backend and frontend code
COPY backend/ backend/
COPY frontend/ frontend/

# 6. Set workdir to backend
WORKDIR /app/backend

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Run FastAPI with uvicorn
CMD ["uvicorn", "fastapi_ndvi_etc_api:app", "--host", "0.0.0.0", "--port", "8000"]
