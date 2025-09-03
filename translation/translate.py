import os
import glob
import re
import textwrap
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def split_into_sentences(text: str) -> list[str]:
    """Splits a block of text into a list of sentences."""
    text = textwrap.dedent(text).strip()
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r'([.!?।])\s*', text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i].strip() + parts[i+1])
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences

# NLLB-200 model initialization
print("🔄 Loading NLLB-200 model... (this may take a few minutes on first run)")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ NLLB-200 model loaded successfully.\n")

# Language codes mapping
LANGUAGE_CODES = {
    'hindi': 'hin_Deva',
    'bengali': 'ben_Beng',
    'tamil': 'tam_Taml',
    'telugu': 'tel_Telu',
    'marathi': 'mar_Deva',
    'gujarati': 'guj_Gujr',
    'kannada': 'kan_Knda',
    'malayalam': 'mal_Mlym',
    'punjabi': 'pan_Guru',
    'urdu': 'urd_Arab',
    'french': 'fra_Latn',
    'spanish': 'spa_Latn',
    'german': 'deu_Latn',
    'chinese': 'zho_Hans',
    'japanese': 'jpn_Jpan',
    'korean': 'kor_Hang',
    'arabic': 'arb_Arab',
    'russian': 'rus_Cyrl'
}

def translate_with_nllb(text: str, target_lang: str = "hin_Deva") -> str:
    """Translate input text to target language using NLLB-200."""
    if not text.strip():
        return ""
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    forced_bos_token_id = tokenizer.lang_code_to_id.get(target_lang, None)
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
    
    translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translated_text.strip()

def translate_long_text_en_to_hi(text: str, target_language: str = "hindi") -> str:
    """Translates a long English text to specified language by processing it sentence by sentence."""
    target_lang_code = LANGUAGE_CODES.get(target_language.lower(), 'hin_Deva')
    sents = split_into_sentences(text)
    translated_sents = []
    
    for sent in sents:
        if sent.strip():
            translated = translate_with_nllb(sent, target_lang_code)
            translated_sents.append(translated)
    
    return " ".join(translated_sents)

def main():
    # Define input and output folders
    input_folder = "/app/transcripts"
    output_folder = "/app/nllb_translated_texts"
    os.makedirs(output_folder, exist_ok=True)

    # Find all .txt files in the input folder
    transcript_files = sorted(glob.glob(os.path.join(input_folder, "*.txt")))

    if not transcript_files:
        print(f"⚠️ Error: No transcript files found in the '{input_folder}' directory.")
        return

    print(f"--- Found {len(transcript_files)} transcript(s). Starting translation with NLLB-200... ---")

    for file_path in transcript_files:
        filename = os.path.basename(file_path)
        print(f"\n--- Translating: {filename} ---")

        with open(file_path, "r", encoding="utf-8") as f:
            english_text = f.read()
        print(f"   English Text: {english_text}")

        final_hindi = translate_long_text_en_to_hi(english_text, "hindi")
        print(f"   🇮🇳 NLLB-200 Hindi: {final_hindi}")

        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_hindi)
        print(f"   💾 Saved to: {output_path}")

    print("\n\n✅ All text files have been translated successfully using NLLB-200.")

if __name__ == "__main__":
    main()