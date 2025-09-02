import os
import glob
import whisper
from gtts import gTTS

def transcribe_english(model, audio_path: str) -> str:
    """
    Transcribe English audio into English text using the loaded Whisper model.
    """
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language="en",
        fp16=False,
        verbose=False,
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    )
    return result["text"]

def main():
    # Load the Whisper model
    print("\n🧠 Loading the Whisper 'medium' model... (This may take a moment)")
    model = whisper.load_model("medium")
    print("✅ Model loaded successfully.")

    # Define input and output folders for Docker environment
    input_folder = "/app/input"
    output_folder = "/app/output"
    os.makedirs(output_folder, exist_ok=True)

    # Get a sorted list of all .wav files in the input folder
    audio_files = sorted(glob.glob(os.path.join(input_folder, "*.wav")))

    if not audio_files:
        print("⚠️ No audio files found in the input folder.")
        return

    print(f"\nFound {len(audio_files)} audio files. Starting transcription process...")

    # Process each audio file
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        print(f"\n--- Processing: {filename} ---")

        # Transcribe the audio chunk
        english_text = transcribe_english(model, audio_path)
        print(f"📝 Transcription: {english_text}")

        # Save the transcription
        base_filename = os.path.splitext(filename)[0]
        transcript_path = os.path.join(output_folder, f"{base_filename}.txt")

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(english_text)
        print(f"💾 Saved transcript to: {transcript_path}")

    print("\n\n✅ All chunks have been transcribed and saved.")
    print("\nContents of the output folder:")
    print("\n".join(os.listdir(output_folder)))

if __name__ == "__main__":
    main()