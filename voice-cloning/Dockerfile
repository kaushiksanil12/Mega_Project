FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy the voice cloning script
COPY voice_clone.py .

# Create necessary directories
RUN mkdir -p /app/final_chunks /app/google_translated_texts /app/cloned_hindi_audio_soft /app/temp_pitch_shifted

# Set environment variable
ENV COQUI_TOS_AGREED=1

# Set the entry point
ENTRYPOINT ["python", "voice_clone.py"]
