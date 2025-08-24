FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create logs directory
RUN mkdir -p /app/logs

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port for health checks
EXPOSE 5000

# Create a non-root user for security
RUN addgroup --system --gid 1001 appuser && \
    adduser --system --uid 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Run the application
CMD ["python", "main.py"]