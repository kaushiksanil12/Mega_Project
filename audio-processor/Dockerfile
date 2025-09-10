# Use Python 3.11 slim with multimedia support
FROM python:3.11-slim

# Install system dependencies for multimedia processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p input output output/chunks

# Copy application code
COPY audio_processor.py ./

# Set permissions
RUN chmod +x audio_processor.py

# Run as non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Default command
CMD ["python", "audio_processor.py"]
