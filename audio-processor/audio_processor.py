import os
import sys
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa
import noisereduce as nr
import numpy as np
import wave
import subprocess

def convert_to_wav(input_file):
    """Convert any audio format to WAV using pydub"""
    try:
        sound = AudioSegment.from_file(input_file)
        wav_path = os.path.splitext(input_file)[0] + "_converted.wav"
        sound.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

def separate_vocals(input_file):
    """Separate vocals from background music using Spleeter"""
    try:
        base_dir = os.getcwd()
        output_dir = os.path.join(base_dir, "separated_audio")
        
        # Run spleeter command using subprocess
        print(f"Running Spleeter... Output will be in '{output_dir}'")
        cmd = ["spleeter", "separate", "-p", "spleeter:2stems-16kHz", "-o", output_dir, input_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Spleeter error: {result.stderr}")
            return None
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        vocals_path = os.path.join(output_dir, base_name, "vocals.wav")
        
        if not os.path.exists(vocals_path):
            print(f"Error: Vocals file not found at {vocals_path}")
            return None
            
        return vocals_path
    except Exception as e:
        print(f"Error separating vocals: {e}")
        return None

def denoise_audio(input_file):
    """Apply noise reduction and save processed audio"""
    try:
        y, sr = librosa.load(input_file, sr=None)
        
        # Perform noise reduction
        reduced_noise = nr.reduce_noise(y=y, sr=sr)
        
        # Save denoised audio
        denoised_path = os.path.splitext(input_file)[0] + "_denoised.wav"
        with wave.open(denoised_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((reduced_noise * 32767).astype(np.int16).tobytes())
        return denoised_path
    except Exception as e:
        print(f"Error denoising audio: {e}")
        return None

def split_audio_into_chunks(audio_path, output_dir, target_chunk_length_ms=10000):
    """Splits an audio file into chunks of a target length based on silence."""
    try:
        print(f"   -> Loading audio from: {audio_path}")
        audio_segment = AudioSegment.from_wav(audio_path)
        
        # Split audio based on silence
        silence_chunks = split_on_silence(
            audio_segment,
            min_silence_len=700,
            silence_thresh=audio_segment.dBFS - 16,
            keep_silence=300
        )
        
        if not silence_chunks:
            print("Could not find any silence to split the audio. Creating single chunk.")
            silence_chunks = [audio_segment]
        
        # Merge small chunks to be approximately the target length
        merged_chunks = []
        current_chunk = AudioSegment.empty()
        
        for chunk in silence_chunks:
            current_chunk += chunk
            if len(current_chunk) >= target_chunk_length_ms:
                merged_chunks.append(current_chunk)
                current_chunk = AudioSegment.empty()
        
        # Add the last remaining chunk if it exists
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        # Export the final chunks
        print(f"   -> Exporting {len(merged_chunks)} chunks to '{output_dir}'")
        for i, chunk in enumerate(merged_chunks):
            chunk_name = f"chunk_{i+1:03d}.wav"
            chunk_path = os.path.join(output_dir, chunk_name)
            chunk.export(chunk_path, format="wav")
        
        return len(merged_chunks)
    except Exception as e:
        print(f"Error splitting audio: {e}")
        return 0

def process_audio(input_file):
    """Main processing function"""
    # Convert to WAV if needed
    if not input_file.lower().endswith('.wav'):
        print("Converting to WAV...")
        wav_file = convert_to_wav(input_file)
        if not wav_file:
            return None, None
    else:
        wav_file = input_file
    
    # Separate vocals
    print("\nSeparating vocals...")
    vocals_path = separate_vocals(wav_file)
    if not vocals_path:
        return None, None
    
    # Denoise
    print("\nDenoising audio...")
    final_audio_path = denoise_audio(vocals_path)
    if not final_audio_path:
        return None, None
        
    print(f"Denoised vocal track saved to: {final_audio_path}")
    
    # Split the final audio
    print("\nSplitting final audio into chunks...")
    chunks_dir = "final_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    chunk_count = split_audio_into_chunks(final_audio_path, chunks_dir)
    
    if chunk_count > 0:
        return final_audio_path, chunks_dir
    else:
        return final_audio_path, None

def main():
    parser = argparse.ArgumentParser(description='Process audio files: separate vocals, denoise, and split into chunks')
    parser.add_argument('input_file', help='Path to input audio file')
    parser.add_argument('--output-dir', default='/app/output', help='Output directory for processed files')
    parser.add_argument('--chunk-length', type=int, default=10000, help='Target chunk length in milliseconds (default: 10000)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Change to output directory for processing
    original_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    try:
        # Process audio
        final_audio_path, chunks_folder = process_audio(args.input_file)
        
        if final_audio_path:
            print("\n--- All Processing Complete ---")
            print(f"Full denoised vocal track saved at: {final_audio_path}")
            if chunks_folder:
                print(f"Split audio chunks are in the folder: {chunks_folder}")
            else:
                print("Audio splitting was skipped due to errors.")
        else:
            print("\n--- Processing Failed ---")
            print("Could not complete audio processing. Check error messages above.")
            sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()
