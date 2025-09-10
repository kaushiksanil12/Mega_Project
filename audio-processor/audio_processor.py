#!/usr/bin/env python3
"""
Audio Processing Script for Dubbing with Denoising
Processes audio from any video/audio file, denoises it, then creates WAV chunks
"""

import os
import warnings
from typing import Final, Mapping, Sequence, Optional
import glob

import numpy as np
import torch
from moviepy.editor import VideoFileClip, AudioFileClip
from pyannote.audio import Pipeline
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf

# Configuration
INPUT_DIR = "input"
OUTPUT_DIR = "output"
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")
DENOISED_DIR = os.path.join(OUTPUT_DIR, "denoised")  # NEW: Denoised audio directory

# Supported file extensions
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.3gp']
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma']
ALL_EXTENSIONS = VIDEO_EXTENSIONS + AUDIO_EXTENSIONS

# Create directories if they don't exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(DENOISED_DIR, exist_ok=True)  # NEW: Create denoised directory

# Disable Hugging Face token requirement
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

def log_info(message):
    """Simple logging function"""
    print(f"[INFO] {message}")

def log_error(message):
    """Simple error logging function"""
    print(f"[ERROR] {message}")

def find_media_files() -> list:
    """Find all video and audio files in the input directory."""
    media_files = []
    
    for ext in ALL_EXTENSIONS:
        pattern = os.path.join(INPUT_DIR, f"*{ext}")
        files = glob.glob(pattern, recursive=False)
        media_files.extend(files)
    
    # Also check uppercase extensions
    for ext in ALL_EXTENSIONS:
        pattern = os.path.join(INPUT_DIR, f"*{ext.upper()}")
        files = glob.glob(pattern, recursive=False)
        media_files.extend(files)
    
    # Remove duplicates and sort
    media_files = sorted(list(set(media_files)))
    
    log_info(f"Found {len(media_files)} media files: {[os.path.basename(f) for f in media_files]}")
    return media_files

def extract_audio_from_file(file_path: str) -> str:
    """Extract audio from video file or copy audio file."""
    file_ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    audio_output_path = os.path.join(INPUT_DIR, f"{base_name}_extracted_audio.mp3")
    
    try:
        if file_ext in VIDEO_EXTENSIONS:
            log_info(f"Extracting audio from video: {os.path.basename(file_path)}")
            # Extract audio from video
            clip = VideoFileClip(file_path)
            if clip.audio is None:
                raise ValueError("Video file has no audio track")
            
            clip.audio.write_audiofile(audio_output_path, verbose=False, logger=None)
            clip.close()
            
        elif file_ext in AUDIO_EXTENSIONS:
            log_info(f"Processing audio file: {os.path.basename(file_path)}")
            # Convert audio file to MP3 if needed
            if file_ext == '.mp3':
                # If it's already MP3, just use the original
                return file_path
            else:
                # Convert to MP3
                audio = AudioSegment.from_file(file_path)
                audio.export(audio_output_path, format="mp3")
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        log_info(f"Audio extracted to: {audio_output_path}")
        return audio_output_path
    
    except Exception as e:
        log_error(f"Failed to extract audio from {file_path}: {e}")
        return None

def denoise_audio(input_file: str, noise_reduce_factor: float = 0.8) -> str:
    """
    FIXED: Denoise audio file using correct noisereduce API.
    
    Args:
        input_file: Path to input audio file
        noise_reduce_factor: How much noise to reduce (0.0 to 1.0)
    
    Returns:
        Path to denoised audio file
    """
    try:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        denoised_output_path = os.path.join(DENOISED_DIR, f"{base_name}_denoised.wav")
        
        log_info(f"🔇 Denoising audio: {os.path.basename(input_file)}")
        
        # Load audio file using soundfile for better quality
        audio_data, sample_rate = sf.read(input_file)
        
        # Handle stereo audio by converting to mono for processing
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Use first 0.5 seconds as noise sample (or 10% of audio, whichever is smaller)
        noise_duration = min(0.5, len(audio_data) / sample_rate * 0.1)
        noise_sample_length = int(noise_duration * sample_rate)
        noise_sample = audio_data[:noise_sample_length]
        
        # FIXED: Use correct noisereduce API parameters
        denoised_audio = nr.reduce_noise(
            y=audio_data,                    # Changed from audio_clip
            y_noise=noise_sample,            # Changed from noise_clip  
            sr=sample_rate,                  # Sample rate
            prop_decrease=noise_reduce_factor,
            stationary=False                 # Keep this - it works
        )
        
        # Save denoised audio as WAV for best quality
        sf.write(denoised_output_path, denoised_audio, sample_rate)
        
        log_info(f"✅ Denoised audio saved: {os.path.basename(denoised_output_path)}")
        return denoised_output_path
        
    except Exception as e:
        log_error(f"❌ Denoising failed for {input_file}: {e}")
        # Fallback to simple pydub filtering
        return fallback_audio_cleanup(input_file)

def fallback_audio_cleanup(input_file: str) -> str:
    """
    NEW FUNCTION: Fallback audio cleanup using pydub if noisereduce fails.
    """
    try:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        cleaned_output_path = os.path.join(DENOISED_DIR, f"{base_name}_cleaned.wav")
        
        log_info(f"🔧 Applying basic audio cleanup: {os.path.basename(input_file)}")
        
        # Load with pydub
        audio = AudioSegment.from_file(input_file)
        
        # Apply high-pass filter to remove low-frequency noise (below 80Hz)
        filtered_audio = audio.high_pass_filter(80)
        
        # Apply low-pass filter to remove high-frequency noise (above 8kHz for speech)
        filtered_audio = filtered_audio.low_pass_filter(8000)
        
        # Normalize audio to consistent levels
        normalized_audio = filtered_audio.normalize()
        
        # Export as WAV
        normalized_audio.export(cleaned_output_path, format="wav")
        
        log_info(f"✅ Cleaned audio saved: {os.path.basename(cleaned_output_path)}")
        return cleaned_output_path
        
    except Exception as e:
        log_error(f"❌ Audio cleanup also failed: {e}")
        return input_file

def create_pyannote_timestamps(audio_file: str, device: str = "cpu") -> Sequence[Mapping[str, float]]:
    """Creates timestamps using PyAnnote speaker diarization without HF token."""
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=False
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if device == "cuda":
                pipeline.to(torch.device("cuda"))
            
            diarization = pipeline(audio_file)
            utterance_metadata = [
                {"start": segment.start, "end": segment.end, "speaker_id": speaker}
                for segment, _, speaker in diarization.itertracks(yield_label=True)
            ]
            log_info(f"Created {len(utterance_metadata)} speaker segments")
            return utterance_metadata
    
    except Exception as e:
        log_error(f"PyAnnote failed: {e}. Using simple segmentation.")
        return create_simple_timestamps(audio_file)

def create_simple_timestamps(audio_file: str, segment_duration: float = 5.0) -> Sequence[Mapping[str, float]]:
    """Fallback: Creates simple time-based segments."""
    try:
        # Try with moviepy first
        try:
            clip = AudioFileClip(audio_file)
            duration = clip.duration
            clip.close()
        except:
            # Fallback to pydub
            audio = AudioSegment.from_file(audio_file)
            duration = len(audio) / 1000.0  # Convert ms to seconds
        
        segments = []
        start = 0.0
        segment_id = 0
        
        while start < duration:
            end = min(start + segment_duration, duration)
            segments.append({
                "start": start,
                "end": end,
                "speaker_id": f"speaker_{segment_id % 2}"
            })
            start = end
            segment_id += 1
        
        log_info(f"Created {len(segments)} simple time segments")
        return segments
    
    except Exception as e:
        log_error(f"Simple segmentation failed: {e}")
        return []

def cut_and_save_audio(audio_file: str, utterance_metadata: Sequence[Mapping[str, float]]) -> Sequence[Mapping[str, str]]:
    """Cuts DENOISED audio into chunks and saves them in WAV format."""
    audio = AudioSegment.from_file(audio_file)
    updated_metadata = []
    
    for i, utterance in enumerate(utterance_metadata):
        start_ms = int(utterance["start"] * 1000)
        end_ms = int(utterance["end"] * 1000)
        chunk = audio[start_ms:end_ms]
        
        # Save as WAV format with high quality
        chunk_filename = f"chunk_{utterance['start']:.1f}_{utterance['end']:.1f}.wav"
        chunk_path = os.path.join(CHUNKS_DIR, chunk_filename)
        
        # Export as WAV with high quality settings
        chunk.export(
            chunk_path, 
            format="wav",
            parameters=["-ac", "2", "-ar", "44100"]  # Stereo, 44.1kHz
        )
        
        utterance_copy = utterance.copy()
        utterance_copy["path"] = chunk_path
        updated_metadata.append(utterance_copy)
        
        log_info(f"Saved clean WAV chunk {i+1}: {chunk_filename}")
    
    return updated_metadata

def insert_audio_at_timestamps(utterance_metadata: Sequence[Mapping], background_audio_file: str, output_filename: str = "dubbed_vocals.mp3") -> str:
    """Inserts dubbed audio chunks at specified timestamps."""
    background = AudioSegment.from_file(background_audio_file)
    total_duration = len(background)
    output_audio = AudioSegment.silent(duration=total_duration)
    
    for item in utterance_metadata:
        try:
            # Skip if not marked for dubbing
            if not item.get("for_dubbing", True):
                continue
            
            start_time = int(item["start"] * 1000)
            dubbed_path = item.get("dubbed_path", item.get("path"))
            
            if dubbed_path and os.path.exists(dubbed_path):
                # Load WAV file
                audio_chunk = AudioSegment.from_wav(dubbed_path)
                output_audio = output_audio.overlay(audio_chunk, position=start_time)
                log_info(f"Inserted clean WAV audio at {item['start']:.1f}s")
        
        except Exception as e:
            log_error(f"Failed to insert audio: {e}")
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    output_audio.export(output_path, format="mp3")
    log_info(f"Saved dubbed vocals: {output_path}")
    return output_path

def merge_background_and_vocals(background_file: str, vocals_file: str, target_language: str = "en", original_filename: str = "") -> str:
    """Merges background audio with dubbed vocals."""
    try:
        background = AudioSegment.from_file(background_file)
        vocals = AudioSegment.from_file(vocals_file)
        
        # Normalize and adjust volumes
        background = background.normalize()
        vocals = vocals.normalize() + 5  # Boost vocals by 5dB
        
        # Match lengths
        min_length = min(len(background), len(vocals))
        background = background[:min_length]
        vocals = vocals[:min_length]
        
        # Mix audio
        mixed_audio = background.overlay(vocals)
        
        # Use original filename if provided
        if original_filename:
            base_name = os.path.splitext(original_filename)[0]
            output_filename = f"{base_name}_denoised_dubbed_{target_language}.mp3"
        else:
            output_filename = f"denoised_dubbed_audio_{target_language}.mp3"
        
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        mixed_audio.export(output_path, format="mp3")
        
        log_info(f"Final denoised dubbed audio saved: {output_path}")
        return output_path
    
    except Exception as e:
        log_error(f"Audio merging failed: {e}")
        return ""

def process_single_file(file_path: str, target_language: str = "en"):
    """Process a single media file with denoising."""
    log_info(f"\n🎬 Processing: {os.path.basename(file_path)}")
    
    # Step 1: Extract audio from video/audio file
    audio_file = extract_audio_from_file(file_path)
    if not audio_file:
        return
    
    # Step 2: DENOISE THE AUDIO (NEW STEP)
    log_info("🔇 Denoising audio...")
    denoised_audio_file = denoise_audio(audio_file)
    
    # Step 3: Create speaker timestamps from denoised audio
    log_info("📊 Creating speaker timestamps from clean audio...")
    timestamps = create_pyannote_timestamps(denoised_audio_file)
    
    if not timestamps:
        log_error("Failed to create timestamps")
        return
    
    # Step 4: Cut DENOISED audio into WAV chunks
    log_info("✂️ Cutting clean audio into WAV chunks...")
    chunk_metadata = cut_and_save_audio(denoised_audio_file, timestamps)
    
    # Step 5: Process chunks (here you would add your dubbing logic)
    log_info("🎵 Processing clean WAV chunks for dubbing...")
    for chunk in chunk_metadata:
        chunk["for_dubbing"] = True  # Mark all for dubbing
        chunk["dubbed_path"] = chunk["path"]  # Use original for now
    
    # Step 6: Insert dubbed audio at timestamps
    log_info("🔀 Inserting dubbed clean audio...")
    vocals_path = insert_audio_at_timestamps(chunk_metadata, denoised_audio_file)
    
    # Step 7: Create final output
    original_filename = os.path.basename(file_path)
    final_path = merge_background_and_vocals(denoised_audio_file, vocals_path, target_language, original_filename)
    
    return final_path

def main():
    """Main processing function."""
    print("=== Audio/Video Processor with Denoising + WAV Chunks ===")
    
    # Find all media files in input directory
    media_files = find_media_files()
    
    if not media_files:
        log_error(f"No media files found in '{INPUT_DIR}' directory")
        log_info(f"Supported formats: {', '.join(ALL_EXTENSIONS)}")
        return
    
    # Process each file
    processed_files = []
    for file_path in media_files:
        try:
            result = process_single_file(file_path, target_language="en")
            if result:
                processed_files.append(result)
        except Exception as e:
            log_error(f"Failed to process {file_path}: {e}")
    
    print(f"\n✅ Processing complete!")
    print(f"📁 Processed {len(processed_files)} files:")
    for file_path in processed_files:
        print(f"   - {os.path.basename(file_path)}")
    
    if processed_files:
        print(f"📂 Output directory: {os.path.abspath(OUTPUT_DIR)}")
        print(f"🔇 Denoised audio: {os.path.abspath(DENOISED_DIR)}")
        print(f"🎵 Clean WAV chunks: {os.path.abspath(CHUNKS_DIR)}")

if __name__ == "__main__":
    main()
