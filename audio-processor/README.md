# Audio Processor Docker

This Docker container processes audio files to separate vocals, apply noise reduction, and split into chunks using Spleeter.

## Setup

1. Clone or create this directory structure
2. Build the image: `docker build -t audio-processor .`
3. Create input/output directories: `mkdir -p input output`

## Usage

### Method 1: Direct Docker Run
