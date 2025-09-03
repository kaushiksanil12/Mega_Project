FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the translation script
COPY translate.py .

# Create input and output directories
RUN mkdir -p /app/transcripts /app/nllb_translated_texts

# Set the entry point
ENTRYPOINT ["python", "translate.py"]