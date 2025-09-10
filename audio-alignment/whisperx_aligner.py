from pydub import AudioSegment
import whisperx
import librosa
import soundfile as sf
import os
import glob
import tempfile

class WhisperXAligner:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"🎤 Loading WhisperX model on {device}...")
        
        # Load WhisperX model
        self.asr_model = whisperx.load_model("base", device=device)
        
        # Load alignment model  
        model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
        self.align_model = model_a
        self.align_metadata = metadata
        
    def transcribe_and_align(self, audio_path):
        """
        Transcribe audio and get word-level timestamps using WhisperX
        """
        print(f"🎯 Transcribing and aligning: {os.path.basename(audio_path)}")
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe with WhisperX
        result = self.asr_model.transcribe(audio, batch_size=16)
        
        # Perform alignment to get word-level timestamps
        aligned_result = whisperx.align(
            result["segments"], 
            self.align_model, 
            self.align_metadata, 
            audio, 
            self.device, 
            return_char_alignments=False
        )
        
        return aligned_result
    
    def extract_sentence_boundaries(self, aligned_result):
        """
        Extract sentence boundaries and natural pause locations
        """
        print("📝 Extracting sentence boundaries...")
        
        boundaries = []
        
        for segment in aligned_result["segments"]:
            # Get segment text and timing
            text = segment.get("text", "").strip()
            start_time = segment.get("start", 0) * 1000  # Convert to ms
            end_time = segment.get("end", 0) * 1000
            
            # Check if this is end of sentence (contains punctuation)
            is_sentence_end = any(punct in text for punct in ['.', '!', '?', ':', ';'])
            
            # Get word-level boundaries within segment
            if "words" in segment:
                for word in segment["words"]:
                    word_start = word.get("start", 0) * 1000
                    word_end = word.get("end", 0) * 1000
                    word_text = word.get("word", "").strip()
                    
                    # Check if word ends with punctuation
                    word_ends_sentence = any(punct in word_text for punct in ['.', '!', '?'])
                    
                    boundaries.append({
                        'start': word_start,
                        'end': word_end,
                        'text': word_text,
                        'is_sentence_end': word_ends_sentence,
                        'segment_end': word_end == end_time
                    })
            
            # Mark segment boundaries
            boundaries.append({
                'start': start_time,
                'end': end_time,
                'text': text,
                'is_sentence_end': is_sentence_end,
                'is_segment_boundary': True
            })
        
        # Filter for sentence boundaries
        sentence_ends = [b for b in boundaries if b.get('is_sentence_end') or b.get('is_segment_boundary')]
        
        print(f"  📊 Found {len(sentence_ends)} natural sentence/phrase boundaries")
        return sentence_ends
    
    def create_natural_alignment(self, cloned_chunks_dir, original_chunks_dir, 
                                original_full_audio_path, output_path):
        """
        Create alignment with natural sentence-based pauses
        """
        print("=== Natural Sentence-Based Audio Alignment ===")
        print("🎯 Using WhisperX for precise sentence boundary detection")
        
        # Step 1: Get sentence boundaries from original audio
        sentence_boundaries = self.transcribe_and_align(original_full_audio_path)
        natural_breaks = self.extract_sentence_boundaries(sentence_boundaries)
        
        # Step 2: Load and stretch chunks to match originals
        stretched_chunks = self.stretch_chunks_to_originals(cloned_chunks_dir, original_chunks_dir)
        
        # Step 3: Combine chunks
        combined_audio = AudioSegment.empty()
        for chunk in stretched_chunks:
            combined_audio += chunk
        
        # Step 4: Insert pauses at natural sentence boundaries
        final_audio = self.insert_natural_pauses(combined_audio, natural_breaks, original_full_audio_path)
        
        # Export result
        final_audio.export(output_path, format="wav")
        
        print(f"\n🎉 Natural alignment completed!")
        print(f"📁 Output: {output_path}")
        return output_path
    
    def stretch_chunks_to_originals(self, cloned_chunks_dir, original_chunks_dir):
        """
        Stretch cloned chunks to match original chunk durations
        """
        print("\n🔧 Stretching cloned chunks...")
        
        original_files = sorted(glob.glob(os.path.join(original_chunks_dir, "chunk_*.wav")))
        cloned_files = sorted(glob.glob(os.path.join(cloned_chunks_dir, "chunk_*.wav")))
        
        stretched_chunks = []
        
        for orig_file, cloned_file in zip(original_files, cloned_files):
            orig_chunk = AudioSegment.from_file(orig_file)
            target_duration = len(orig_chunk)
            
            # High-quality stretch using librosa
            stretched_chunk = self.high_quality_stretch(cloned_file, target_duration)
            stretched_chunks.append(stretched_chunk)
        
        return stretched_chunks
    
    def high_quality_stretch(self, cloned_file, target_duration_ms):
        """
        High-quality time stretching with librosa
        """
        cloned_chunk = AudioSegment.from_file(cloned_file)
        current_duration = len(cloned_chunk)
        
        if abs(current_duration - target_duration_ms) < 50:  # Less than 50ms diff
            return cloned_chunk
        
        # Calculate stretch ratio
        stretch_ratio = current_duration / target_duration_ms
        
        try:
            # Load with librosa
            y, sr = librosa.load(cloned_file, sr=None)
            
            # Apply time stretch with pitch preservation
            y_stretched = librosa.effects.time_stretch(y, rate=stretch_ratio)
            
            # Convert back to AudioSegment
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
            print(f"   ⚠️ Stretch failed, using original: {e}")
            return cloned_chunk
    
    def insert_natural_pauses(self, combined_audio, sentence_boundaries, original_path):
        """
        Insert pauses at natural sentence boundaries
        """
        print("\n🔇 Inserting natural pauses at sentence boundaries...")
        
        original_audio = AudioSegment.from_file(original_path)
        final_audio = AudioSegment.empty()
        
        last_pos = 0
        
        for boundary in sentence_boundaries:
            if boundary.get('is_sentence_end') or boundary.get('is_segment_boundary'):
                boundary_time = boundary['end']  # ms
                
                # Calculate position in combined audio
                position_ratio = boundary_time / len(original_audio)
                combined_position = int(position_ratio * len(combined_audio))
                
                # Add audio up to boundary
                if combined_position > last_pos:
                    segment = combined_audio[last_pos:combined_position]
                    final_audio += segment
                
                # Add natural pause (300-800ms depending on punctuation)
                text = boundary.get('text', '')
                if '.' in text or '!' in text or '?' in text:
                    pause_duration = 600  # Longer pause for sentence end
                elif ',' in text or ':' in text or ';' in text:
                    pause_duration = 300  # Shorter pause for phrases
                else:
                    pause_duration = 200  # Minimal pause
                
                final_audio += AudioSegment.silent(duration=pause_duration)
                last_pos = combined_position
                
                print(f"    🔇 Added {pause_duration}ms pause after: '{text[:30]}...'")
        
        # Add remaining audio
        if last_pos < len(combined_audio):
            final_audio += combined_audio[last_pos:]
        
        return final_audio

def main():
    print("=== WhisperX Natural Alignment Service ===")
    
    # Input directories
    cloned_chunks_dir = "/app/cloned_chunks"
    original_chunks_dir = "/app/original_chunks"
    original_full_path = "/app/original_full"
    output_dir = "/app/output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find original full audio
    original_full_files = glob.glob(os.path.join(original_full_path, "*_denoised.wav"))
    if not original_full_files:
        original_full_files = glob.glob(os.path.join(original_full_path, "*.wav"))
    
    if not original_full_files:
        print(f"❌ No original full audio found")
        return
    
    original_full_audio = original_full_files[0]
    print(f"✅ Using: {os.path.basename(original_full_audio)}")
    
    # Create WhisperX aligner
    aligner = WhisperXAligner(device="cpu")  # Use "cuda" if GPU available
    
    # Perform natural alignment
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
