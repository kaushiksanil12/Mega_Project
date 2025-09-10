import os
import glob
import re
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from typing import List, Tuple

def parse_timestamp_from_filename(filename: str) -> Tuple[float, float]:
    """
    Extract start and end timestamps from filename like 'hindi_chunk_103.2_104.2.wav'
    Returns: (start_time, end_time) in seconds
    """
    match = re.search(r'(\d+\.\d+)_(\d+\.\d+)', filename)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return start_time, end_time
    else:
        raise ValueError(f"Could not parse timestamp from filename: {filename}")

def time_stretch_audio(input_path: str, target_duration: float, output_path: str):
    """
    Time-stretch audio using librosa to match target duration without changing pitch.
    
    Args:
        input_path: Path to input audio file
        target_duration: Target duration in seconds
        output_path: Path to save stretched audio
    """
    # Load audio
    y, sr = librosa.load(input_path, sr=None)
    current_duration = len(y) / sr
    
    # Calculate stretch rate
    stretch_rate = current_duration / target_duration
    
    if abs(stretch_rate - 1.0) < 0.01:  # No significant stretch needed
        # Just copy the file
        sf.write(output_path, y, sr)
        return
    
    print(f"   -> Stretching from {current_duration:.2f}s to {target_duration:.2f}s (rate: {stretch_rate:.2f})")
    
    # Time stretch using librosa
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
    
    # Save stretched audio
    sf.write(output_path, y_stretched, sr)

def combine_cloned_chunks(
    cloned_chunks_folder: str, 
    output_combined_path: str,
    temp_folder: str = "./temp_stretched"
):
    """
    Combine cloned voice chunks into a single audio file, time-stretching each chunk
    to match its original timestamp duration.
    
    Args:
        cloned_chunks_folder: Folder containing cloned voice chunks
        output_combined_path: Path to save final combined audio
        temp_folder: Temporary folder for stretched chunks
    """
    
    # Create temp folder
    os.makedirs(temp_folder, exist_ok=True)
    
    # Find all cloned audio files
    chunk_files = glob.glob(os.path.join(cloned_chunks_folder, "hindi_chunk_*.wav"))
    
    if not chunk_files:
        print("❌ No cloned chunk files found!")
        return
    
    print(f"📁 Found {len(chunk_files)} cloned chunks to combine")
    
    # Parse timestamps and sort by start time
    chunk_data = []
    for chunk_file in chunk_files:
        filename = os.path.basename(chunk_file)
        try:
            start_time, end_time = parse_timestamp_from_filename(filename)
            duration = end_time - start_time
            chunk_data.append({
                'file': chunk_file,
                'filename': filename,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
        except ValueError as e:
            print(f"⚠️ Warning: {e}")
            continue
    
    # Sort chunks by start time
    chunk_data.sort(key=lambda x: x['start_time'])
    
    print(f"🔄 Processing chunks in chronological order...")
    
    # Time-stretch each chunk to match original duration
    stretched_files = []
    for i, chunk_info in enumerate(chunk_data):
        print(f"\n▶️ Processing: {chunk_info['filename']}")
        print(f"   -> Original timestamp: {chunk_info['start_time']:.1f}s - {chunk_info['end_time']:.1f}s")
        print(f"   -> Target duration: {chunk_info['duration']:.2f}s")
        
        # Create stretched version
        stretched_filename = f"stretched_{i:03d}_{chunk_info['filename']}"
        stretched_path = os.path.join(temp_folder, stretched_filename)
        
        time_stretch_audio(
            chunk_info['file'], 
            chunk_info['duration'], 
            stretched_path
        )
        
        stretched_files.append({
            'file': stretched_path,
            'start_time': chunk_info['start_time'],
            'end_time': chunk_info['end_time']
        })
    
    # Combine all stretched chunks with proper timing
    print(f"\n🔗 Combining {len(stretched_files)} stretched chunks...")
    
    combined_audio = AudioSegment.empty()
    last_end_time = 0.0
    
    for i, stretched_info in enumerate(stretched_files):
        chunk_audio = AudioSegment.from_wav(stretched_info['file'])
        current_start = stretched_info['start_time']
        
        # Add silence gap if there's a gap between chunks
        if current_start > last_end_time:
            gap_duration_ms = (current_start - last_end_time) * 1000
            silence = AudioSegment.silent(duration=int(gap_duration_ms))
            combined_audio += silence
            print(f"   -> Added {gap_duration_ms/1000:.2f}s silence gap")
        
        # Add the chunk
        combined_audio += chunk_audio
        last_end_time = stretched_info['end_time']
        
        print(f"   -> Added chunk {i+1}: {current_start:.1f}s - {stretched_info['end_time']:.1f}s")
    
    # Export final combined audio
    print(f"\n💾 Saving combined audio to: {output_combined_path}")
    combined_audio.export(output_combined_path, format="wav")
    
    # Cleanup temp files
    print(f"🧹 Cleaning up temporary files...")
    for stretched_info in stretched_files:
        if os.path.exists(stretched_info['file']):
            os.remove(stretched_info['file'])
    
    # Remove temp folder if empty
    try:
        os.rmdir(temp_folder)
    except OSError:
        pass  # Folder not empty or doesn't exist
    
    print(f"\n✅ Successfully combined {len(chunk_data)} chunks!")
    print(f"📊 Total duration: {len(combined_audio)/1000:.2f} seconds")
    print(f"📁 Output saved to: {output_combined_path}")

def main():
    """Main function to combine cloned voice chunks."""
    
    # Configuration - CORRECTED PATH
    cloned_chunks_folder = "../voice-cloning/output"  # ✅ Correct path to your cloned chunks
    output_file = "./output/final_dubbed_audio.wav"
    
    print("🎵 Audio Chunk Combiner")
    print("=" * 50)
    print(f"📂 Looking for cloned chunks in: {cloned_chunks_folder}")
    print(f"🎯 Output file: {output_file}")
    
    # Check if input folder exists
    if not os.path.exists(cloned_chunks_folder):
        print(f"❌ ERROR: Cloned chunks folder not found: {cloned_chunks_folder}")
        print(f"📁 Make sure your voice-cloning pipeline completed successfully!")
        return
    
    # List available files for debugging
    chunk_files = glob.glob(os.path.join(cloned_chunks_folder, "hindi_chunk_*.wav"))
    print(f"🔍 Found {len(chunk_files)} hindi_chunk files")
    
    if len(chunk_files) > 0:
        print("📋 Sample files found:")
        for f in sorted(chunk_files)[:3]:
            print(f"   • {os.path.basename(f)}")
        if len(chunk_files) > 3:
            print(f"   • ... and {len(chunk_files) - 3} more")
    
    # Combine the chunks
    combine_cloned_chunks(cloned_chunks_folder, output_file)


if __name__ == "__main__":
    main()
