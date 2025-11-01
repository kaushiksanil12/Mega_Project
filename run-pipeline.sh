#!/bin/bash

set -e  # Exit on any error

echo "🎬 Starting Video-Audio Pipeline..."

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Audio Processing (venv)
echo -e "${GREEN}[1/6] 📦 Running audio-processor...${NC}"
cd audio-processor
source kmyenv/Scripts/activate  # Changed: Use 'source' not 'dot space'
python audio_processor.py
deactivate
cd ..

# Step 2: Transcription (Docker)
echo -e "${BLUE}[2/6] 🐳 Running transcription...${NC}"
cd transcription
docker-compose up --build
docker-compose down
cd ..

# Step 3: Translation (Docker)
echo -e "${BLUE}[3/6] 🐳 Running translation...${NC}"
cd translation
docker-compose up --build
docker-compose down
cd ..

# Step 4: Voice Cloning (Docker)
echo -e "${BLUE}[4/6] 🐳 Running voice-cloning...${NC}"
cd voice-cloning
docker-compose up --build
docker-compose down
cd ..

# Step 5: Audio Alignment (venv)
echo -e "${GREEN}[5/6] 📦 Running audio-alignment...${NC}"
cd audio-alignment
source myenv/Scripts/activate
python simple_boundary_aligner.py
deactivate
cd ..

# Step 6: Merging (venv)
echo -e "${GREEN}[6/6] 📦 Running merging...${NC}"
cd merging
source myenv/Scripts/activate
python combine_chunks.py
deactivate
cd ..

cd audio-processor
rm -rf output
mkdir output
cd ..

cd transcription
rm -rf output
mkdir output
cd ..

cd translation
rm -rf output
mkdir output
cd ..

cd voice-cloning
rm -rf output
mkdir output
cd ..

cd audio-alignment
rm -rf output
mkdir output
cd ..

echo -e "${YELLOW}✅ Pipeline completed successfully!${NC}"
echo "📹 Output video: merging/output/final_dubbed_video_moviepy.mp4"
