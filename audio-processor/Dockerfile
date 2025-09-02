FROM deezer/spleeter:3.8-2stems

# Fix Debian Buster repositories
RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list \
    && sed -i 's/security.debian.org/archive.debian.org/g' /etc/apt/sources.list \
    && sed -i '/buster\/updates/d' /etc/apt/sources.list \
    && sed -i '/buster-security/d' /etc/apt/sources.list

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

COPY audio_processor.py .

RUN mkdir -p /app/input /app/output

ENTRYPOINT ["python", "audio_processor.py"]
