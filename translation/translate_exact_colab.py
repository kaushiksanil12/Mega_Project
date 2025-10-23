import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from pathlib import Path
from tqdm import tqdm
import logging
from huggingface_hub import login


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def translate_all_files():

    login(token="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")
    input_folder = "/app/input"
    output_folder = "/app/output"
    
    Path(output_folder).mkdir(exist_ok=True)
    
    # Use EXACT same setup as working Colab code
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    src_lang, tgt_lang = "eng_Latn", "hin_Deva"
    model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
    
    logger.info(f"Loading model on {DEVICE}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)
    
    ip = IndicProcessor(inference=True)
    
    logger.info("Model loaded successfully!")
    
    # Get all text files
    txt_files = list(Path(input_folder).glob("*.txt"))
    logger.info(f"Found {len(txt_files)} files to translate")
    
    for txt_file in txt_files:
        logger.info(f"Processing: {txt_file.name}")
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            continue
            
        # Process in small batches to avoid memory issues
        batch_size = 2  # Smaller batches for stability
        all_translations = []
        
        for i in tqdm(range(0, len(lines), batch_size), desc=f"Translating {txt_file.name}"):
            batch = lines[i:i + batch_size]
            
            try:
                # Use EXACT same method as working Colab code
                processed_batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
                
                # Tokenize
                inputs = tokenizer(
                    processed_batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(DEVICE)
                
                # Use EXACT same generation method as Colab
                with torch.no_grad():
                    # Get encoder outputs
                    encoder_outputs = model.get_encoder()(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"]
                    )
                    
                    # Start with decoder start token
                    decoder_input_ids = torch.full(
                        (inputs["input_ids"].shape[0], 1),
                        model.config.decoder_start_token_id,
                        dtype=torch.long,
                        device=DEVICE
                    )
                    
                    # Generate step by step
                    for _ in range(256):  # max_length
                        outputs = model(
                            encoder_outputs=encoder_outputs,
                            decoder_input_ids=decoder_input_ids,
                            attention_mask=inputs["attention_mask"]
                        )
                        
                        next_token_logits = outputs.logits[:, -1, :]
                        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
                        
                        # Stop if all sequences have generated EOS token
                        if (next_tokens == tokenizer.eos_token_id).all():
                            break
                
                # Decode
                generated_text = tokenizer.batch_decode(
                    decoder_input_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                # Postprocess using IndicTransToolkit (like Colab)
                translations = ip.postprocess_batch(generated_text, lang=tgt_lang)
                all_translations.extend(translations)
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                all_translations.extend([f"[ERROR: {str(e)}]"] * len(batch))
        
        # Write output
        output_file = Path(output_folder) / f"hindi_{txt_file.name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (original, translation) in enumerate(zip(lines, all_translations), 1):
                f.write(f"{translation}\n")
        
        logger.info(f"✅ Completed: {txt_file.name}")
    
    logger.info("🎉 All files translated!")

if __name__ == "__main__":
    translate_all_files()
