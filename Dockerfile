# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the backend code and models
COPY backend/ backend/
COPY models/ models/

# Expose the port (FastAPI default 8000)
EXPOSE 8000

# Command to run the app with Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
