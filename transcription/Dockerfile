FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the transcription script
COPY transcribe.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set the entry point
ENTRYPOINT ["python", "transcribe.py"]