FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git wget build-essential && rm -rf /var/lib/apt/lists/*

# Install exact Colab versions
RUN pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers==4.56.0 sentencepiece==0.2.1 tqdm accelerate

# Install IndicTransToolkit (it DOES work in Colab despite version check)
RUN pip install IndicTransToolkit

COPY translate_exact_colab.py .
RUN mkdir -p /app/input /app/output

ENV PYTHONUNBUFFERED=1
CMD ["python", "translate_exact_colab.py"]
