FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --timeout=300 --retries=5 -r requirements.txt

COPY simple_boundary_aligner.py .

RUN mkdir -p /app/cloned_chunks /app/original_chunks /app/original_full /app/output /app/temp

ENTRYPOINT ["python", "simple_boundary_aligner.py"]
