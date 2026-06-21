"""
Qwen2-Audio Feature Extraction — Urdu Speech AI Pipeline

Loads the state-of-the-art Qwen2-Audio model's encoder to extract hidden state
embeddings (representing cross-lingual phonetic features) from cleaned audio segments,
then computes the native centroid vector.

Usage:
    python src/extract_qwen_features.py
"""

import argparse
import sys
import numpy as np
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from utils import setup_logging, get_data_paths, load_settings

logger = setup_logging("extract_qwen_features", log_file="logs/extract_qwen_features.log")


def extract_qwen_embedding(model, processor, audio_path):
    """Load audio and run Qwen2-Audio's audio_tower to extract the
    final encoder representation, mean-pooled over time.
    """
    try:
        # Load audio at Qwen's target sample rate (16kHz)
        audio, sr = librosa.load(str(audio_path), sr=16000)
        
        # Preprocess using Qwen2-Audio processor
        inputs = processor(audios=audio, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # Pass inputs to Qwen's audio tower (encoder)
            # Output shape: (batch_size, sequence_len, hidden_dim)
            audio_features = model.model.audio_tower(inputs["input_features"])
            
            # Mean-pool over the sequence length dimension (dim 1)
            mean_pooled = audio_features.mean(dim=1).squeeze(0)
            
            # Convert to numpy array
            embedding = mean_pooled.cpu().numpy()
            
        return embedding
    except Exception as e:
        logger.error(f"  Error extracting Qwen features for {audio_path.name}: {e}")
        return None


def run_extraction(model_id="Qwen/Qwen2-Audio-7B-Instruct", use_enhanced=True):
    settings = load_settings()
    paths = get_data_paths(settings)
    project_root = paths["raw_audio"].parent.parent
    
    # Set up input directories (prefer enhanced audio)
    if use_enhanced:
        input_dir = project_root / "data" / "clean_segments"
        if not input_dir.exists() or not list(input_dir.glob("*.wav")):
            logger.warning("No files found in data/clean_segments. Falling back to data/processed.")
            input_dir = paths["processed"]
    else:
        input_dir = paths["processed"]
        
    # Output directory for Qwen embeddings
    output_dir = project_root / "data" / "features" / "qwen_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wav_files = sorted(list(input_dir.glob("*.wav")))
    if not wav_files:
        logger.error(f"No WAV files found in directory: {input_dir}")
        return
        
    logger.info(f"Loading Qwen2-Audio model '{model_id}'...")
    try:
        # Load processor and model in float16 to keep VRAM usage manageable
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info(f"Model loaded successfully on device: {model.device}")
    except Exception as e:
        logger.error(f"Failed to load Qwen2-Audio model: {e}")
        sys.exit(1)
        
    logger.info(f"Extracting Qwen embeddings for {len(wav_files)} files...")
    
    embeddings = []
    
    for fpath in tqdm(wav_files, desc="Extracting Qwen features"):
        out_path = output_dir / f"{fpath.stem}.npy"
        
        # Avoid redundant extraction
        if out_path.exists():
            embedding = np.load(out_path)
            embeddings.append(embedding)
            continue
            
        embedding = extract_qwen_embedding(model, processor, fpath)
        if embedding is not None:
            np.save(out_path, embedding)
            embeddings.append(embedding)
            
    if not embeddings:
        logger.error("No features were extracted successfully.")
        return
        
    # Compute the native speaker centroid
    embeddings_arr = np.array(embeddings)
    centroid = np.mean(embeddings_arr, axis=0)
    
    centroid_path = project_root / "data" / "features" / "urdu_native_qwen_centroid.npy"
    np.save(centroid_path, centroid)
    
    logger.info(f"🎉 Qwen feature extraction complete!")
    logger.info(f"  - Qwen segment embeddings saved to: {output_dir}")
    logger.info(f"  - Native Qwen centroid saved to: {centroid_path}")
    logger.info(f"  - Feature dimension: {centroid.shape[0]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Qwen2-Audio hidden states.")
    p.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct", help="Qwen2-Audio model ID.")
    p.add_argument("--raw", action="store_true", help="Use raw processed segments instead of DeepFilterNet enhanced.")
    args = p.parse_args()
    
    run_extraction(model_id=args.model, use_enhanced=not args.raw)
