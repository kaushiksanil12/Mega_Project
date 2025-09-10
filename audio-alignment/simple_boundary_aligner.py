from pydub import AudioSegment
import librosa
import soundfile as sf
import os
import glob
import tempfile

class SimpleBoundaryAligner:
    def create_natural_alignment(self, cloned_chunks_dir, original_chunks_dir, 
                               original_full_audio_path, output_path):
        """
        Natural alignment using chunk boundaries as sentence breaks
        """
        print("=== Simple Natural Audio Alignment ===")
        print("🎯 Using chunk boundaries as natural sentence breaks")
        
        # Load chunks
        original_files = sorted(glob.glob(os.path.join(original_chunks_dir, "chunk_*.wav")))
        cloned_files = sorted(glob.glob(os.path.join(cloned_chunks_dir, "chunk_*.wav")))
        
        print(f"📄 Found {len(original_files)} original chunks")
        print(f"🎤 Found {len(cloned_files)} cloned chunks")
        
        # Process and combine with natural pauses
        final_audio = AudioSegment.empty()
        
        for i, (orig_file, cloned_file) in enumerate(zip(original_files, cloned_files)):
            orig_chunk = AudioSegment.from_file(orig_file)
            cloned_chunk = AudioSegment.from_file(cloned_file)
            
            # Match duration with high-quality stretching
            target_duration = len(orig_chunk)
            stretched_chunk = self.high_quality_stretch(cloned_file, target_duration)
            
            # Add the chunk
            final_audio += stretched_chunk
            
            # Add natural pause between sentences (except after last)
            if i < len(original_files) - 1:
                pause_duration = 600  # 600ms natural sentence break
                final_audio += AudioSegment.silent(duration=pause_duration)
                print(f"    🔇 Added {pause_duration}ms pause after chunk {i+1}")
        
        # Export result
        final_audio.export(output_path, format="wav")
        
        print(f"\n🎉 Success! Natural sentence-aligned audio created!")  # Your desired message!
        print(f"📁 Output: {output_path}")
        print(f"⏱️ Final duration: {len(final_audio)/1000:.1f} seconds")
        
        return output_path
    
    def high_quality_stretch(self, cloned_file, target_duration_ms):
        """High-quality time stretching using librosa"""
        cloned_chunk = AudioSegment.from_file(cloned_file)
        current_duration = len(cloned_chunk)
        
        if abs(current_duration - target_duration_ms) < 50:
            return cloned_chunk
        
        stretch_ratio = current_duration / target_duration_ms
        stretch_ratio = max(0.5, min(stretch_ratio, 2.0))  # Limit ratios
        
        try:
            y, sr = librosa.load(cloned_file, sr=None)
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_ratio)
            
            temp_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_path, y_stretched, sr)
            result = AudioSegment.from_file(temp_path)
            os.unlink(temp_path)
            
            # Fine-tune duration
            if len(result) > target_duration_ms:
                result = result[:target_duration_ms]
            elif len(result) < target_duration_ms:
                padding = target_duration_ms - len(result)
                result += AudioSegment.silent(duration=padding)
            
            return result
            
        except Exception as e:
            print(f"   ⚠️ Stretch failed, using padding: {e}")
            # Fallback: simple padding/trimming
            if current_duration < target_duration_ms:
                return cloned_chunk + AudioSegment.silent(duration=target_duration_ms - current_duration)
            else:
                return cloned_chunk[:target_duration_ms]

def main():
    print("=== Simple Natural Alignment Service ===")
    
    # Input directories
    cloned_chunks_dir = "/app/cloned_chunks"
    original_chunks_dir = "/app/original_chunks"
    original_full_path = "/app/original_full"
    output_dir = "/app/output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find inputs
    original_full_files = []
    for pattern in ["*_denoised.wav", "*.wav", "*vocals*.wav"]:
        original_full_files.extend(glob.glob(os.path.join(original_full_path, pattern)))
    
    if not original_full_files:
        print(f"❌ No original full audio found")
        return
    
    original_full_audio = original_full_files[0]
    print(f"✅ Using: {os.path.basename(original_full_audio)}")
    
    # Check chunks
    cloned_count = len(glob.glob(os.path.join(cloned_chunks_dir, "chunk_*.wav")))
    original_count = len(glob.glob(os.path.join(original_chunks_dir, "chunk_*.wav")))
    
    print(f"✅ Found {cloned_count} cloned, {original_count} original chunks")
    
    if cloned_count == 0 or original_count == 0:
        print("❌ Missing chunk files")
        return
    
    # Perform alignment
    aligner = SimpleBoundaryAligner()
    output_path = os.path.join(output_dir, "natural_sentence_aligned.wav")
    
    try:
        result = aligner.create_natural_alignment(
            cloned_chunks_dir=cloned_chunks_dir,
            original_chunks_dir=original_chunks_dir,
            original_full_audio_path=original_full_audio,
            output_path=output_path
        )
        
        print(f"\n🎉 Success! Natural sentence-aligned audio created!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
