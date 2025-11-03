FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create logs directory
RUN mkdir -p /app/logs

# Set working directory
WORKDIR /app

# Copy requirements (excluding torch/torchvision/torchaudio as they're pre-installed)
COPY requirements.txt .

# Install only the additional dependencies (PyTorch already included in base image)
RUN pip install --no-cache-dir \
    transformers==4.33.0 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    kafka-python==2.0.2 \
    psycopg2-binary==2.9.7 \
    redis==5.0.1 \
    flask==2.3.2 \
    flask-socketio==5.3.6 \
    psutil==5.9.5 \
    scikit-learn==1.3.0 \
    PyJWT==2.8.0

# Copy source code
COPY . .

# Expose port for health checks
EXPOSE 5000

# Create a non-root user for security
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

# Run the application
CMD ["python", "main.py"]
