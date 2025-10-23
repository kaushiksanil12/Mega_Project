import os
import glob
import re
import importlib
import torch
from TTS.api import TTS
import librosa
import soundfile as sf


def allowlist_and_load_tts(tts_model_name: str, device: str, max_retries: int = 12):
    """
    Attempts to construct TTS(tts_model_name).to(device).
    If a torch/unpickling error reports an unsupported global, this function:
      - parses the module path and class name from the exception message,
      - imports that class,
      - registers it with torch.serialization.add_safe_globals,
      - retries loading.
    This repeats until the model loads or max_retries is exhausted.
    """
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[loader] Attempt {attempt}: constructing TTS('{tts_model_name}') ...")
            tts = TTS(tts_model_name).to(device)
            print("[loader] Model constructed and moved to device successfully.")
            return tts
        except Exception as e:
            last_exception = e
            msg = str(e)
            m = re.search(r'GLOBAL\s+([\w\.]+)\.([A-Za-z_]\w+)\s+was not', msg)
            if not m:
                m2 = re.search(r'Unsupported global:\s*([\w\.]+)\.([A-Za-z_]\w+)', msg)
                m = m2
            if not m:
                print("[loader] Could not parse missing global from exception message.")
                print("[loader] Exception message:\n", msg)
                break

            module_path = m.group(1)
            class_name = m.group(2)
            print(f"[loader] Detected missing global class: {module_path}.{class_name}")

            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
            except Exception as import_exc:
                print(f"[loader] Failed to import {module_path}.{class_name}: {import_exc}")
                print("[loader] You may need to `pip install` or `git clone` the correct package, or add its path to PYTHONPATH.")
                break

            try:
                torch.serialization.add_safe_globals([cls])
                print(f"[loader] Added {module_path}.{class_name} to torch safe globals.")
            except Exception as add_exc:
                print(f"[loader] Failed to add safe global: {add_exc}")
                break

    print("[loader] Failed to load model after retries. Raising last exception.")
    raise last_exception


def find_matching_audio_file(text_base_name, audio_folder):
    """
    Find the corresponding audio file for a text file, handling naming variations.
    Tries multiple naming patterns to find a match.
    """
    # Pattern 1: Exact match (hindi_chunk_103.2_104.2.wav)
    exact_match = os.path.join(audio_folder, f"{text_base_name}.wav")
    if os.path.exists(exact_match):
        return exact_match
    
    # Pattern 2: Remove 'hindi_' prefix (chunk_103.2_104.2.wav)
    no_prefix = text_base_name.replace('hindi_', '')
    no_prefix_match = os.path.join(audio_folder, f"{no_prefix}.wav")
    if os.path.exists(no_prefix_match):
        return no_prefix_match
    
    # Pattern 3: Try with just the timestamp part
    timestamp_match = re.search(r'(\d+\.\d+_\d+\.\d+)', text_base_name)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        timestamp_file = os.path.join(audio_folder, f"chunk_{timestamp}.wav")
        if os.path.exists(timestamp_file):
            return timestamp_file
    
    return None


def lower_pitch(input_path, output_path, pitch_shift_semitones=-2.0):
    """
    Loads an audio file, lowers its pitch, and saves it to a new path.
    A shift of -2.0 makes the voice noticeably lower and softer.
    """
    print(f"   -> Applying pitch shift of {pitch_shift_semitones} semitones...")
    y, sr = librosa.load(input_path, sr=None)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift_semitones)
    sf.write(output_path, y_shifted, sr)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")
    print("--- Loading Coqui XTTS model (this may take a few minutes)... ---")

    tts_model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts = allowlist_and_load_tts(tts_model_name, device)
    print("✅ Model loaded successfully.\n")

    # Your original folder paths
    english_audio_folder = "/app/chunks"
    hindi_text_folder = "/app/google_translated_texts"
    output_folder = "/app/cloned_hindi_audio_soft"
    temp_folder = "/app/temp_pitch_shifted"

    # Ensure output directories exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    # Check if input directories exist
    if not os.path.exists(english_audio_folder):
        print(f"❌ ERROR: Input audio folder not found: {english_audio_folder}")
        return
    
    if not os.path.exists(hindi_text_folder):
        print(f"❌ ERROR: Input text folder not found: {hindi_text_folder}")
        return

    hindi_files = sorted(glob.glob(os.path.join(hindi_text_folder, "*.txt")))
    
    if not hindi_files:
        print(f"❌ No text files found in {hindi_text_folder}")
        return

    # Debug: Show available files
    print("=== DEBUGGING FILE AVAILABILITY ===")
    available_audio_files = glob.glob(os.path.join(english_audio_folder, "*.wav"))
    print(f"Found {len(available_audio_files)} audio files in {english_audio_folder}:")
    for audio_file in sorted(available_audio_files)[:5]:  # Show first 5
        print(f"  📁 {os.path.basename(audio_file)}")
    if len(available_audio_files) > 5:
        print(f"  ... and {len(available_audio_files) - 5} more")

    print(f"\nFound {len(hindi_files)} text files in {hindi_text_folder}:")
    for text_file in hindi_files[:3]:  # Show first 3
        base_name = os.path.splitext(os.path.basename(text_file))[0]
        print(f"  📄 {base_name}.txt")
        matching_audio = find_matching_audio_file(base_name, english_audio_folder)
        if matching_audio:
            print(f"    ✅ Found matching audio: {os.path.basename(matching_audio)}")
        else:
            print(f"    ❌ No matching audio found")
    print("=== END DEBUGGING ===\n")

    print(f"--- Found {len(hindi_files)} text files to process ---")
    print(f"--- Starting voice cloning process with pitch shifting... ---")
    
    processed_count = 0
    skipped_count = 0
    
    for hindi_text_path in hindi_files:
        base_filename = os.path.splitext(os.path.basename(hindi_text_path))[0]
        print(f"\n▶️ Processing: {base_filename}")

        # Use the improved file matching function
        voice_to_clone_path = find_matching_audio_file(base_filename, english_audio_folder)
        
        if not voice_to_clone_path:
            print(f"❌ ERROR: Cannot find matching audio file for: {base_filename}. Skipping.")
            skipped_count += 1
            continue

        print(f"   ✅ Found audio file: {os.path.basename(voice_to_clone_path)}")

        temp_pitched_voice_path = os.path.join(temp_folder, f"{base_filename}_low.wav")
        print(f"   -> Lowering pitch of source voice...")
        
        try:
            lower_pitch(voice_to_clone_path, temp_pitched_voice_path)
        except Exception as e:
            print(f"   ❌ Error during pitch shifting: {e}")
            skipped_count += 1
            continue

        # Read Hindi text
        with open(hindi_text_path, 'r', encoding='utf-8') as f:
            hindi_text = f.read().strip()
        
        if not hindi_text:
            print(f"   ❌ Empty text file: {hindi_text_path}. Skipping.")
            skipped_count += 1
            continue

        print(f"   -> Text to clone: {hindi_text[:100]}..." if len(hindi_text) > 100 else f"   -> Text to clone: {hindi_text}")

        output_path = os.path.join(output_folder, f"{base_filename}.wav")
        print(f"   -> Cloning voice with Hindi text...")
        
        try:
            tts.tts_to_file(
                text=hindi_text,
                file_path=output_path,
                speaker_wav=temp_pitched_voice_path,
                language="hi",

            )
            print(f"   💾 Saved to: {output_path}")
            processed_count += 1
        except Exception as e:
            print(f"   ❌ Error during TTS generation: {e}")
            skipped_count += 1

        # Clean up temporary file
        if os.path.exists(temp_pitched_voice_path):
            os.remove(temp_pitched_voice_path)

    print(f"\n\n✅ Voice cloning completed!")
    print(f"📊 Successfully processed: {processed_count}/{len(hindi_files)} files")
    print(f"⚠️ Skipped files: {skipped_count}/{len(hindi_files)} files")
    print(f"📁 Output directory: {output_folder}")


if __name__ == "__main__":
    main()
