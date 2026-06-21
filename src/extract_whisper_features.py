"""
Whisper Feature Extraction — Urdu Speech AI Pipeline

Loads Whisper's encoder to extract hidden state embeddings (representing
cross-lingual phonetic features) from cleaned audio segments, then
computes the native centroid vector.

Usage:
    python src/extract_whisper_features.py
"""

import argparse
import sys
import numpy as np
import torch
import whisper
from pathlib import Path
from tqdm import tqdm

from utils import setup_logging, get_data_paths, load_settings

logger = setup_logging("extract_whisper_features", log_file="logs/extract_whisper_features.log")


def extract_whisper_embedding(model, audio_path, device):
    """Load audio, compute log-Mel spectrogram, and run the Whisper encoder
    to extract the final hidden layer's mean-pooled representation.
    """
    try:
        # Load audio (Whisper automatically resamples to 16kHz)
        audio = whisper.load_audio(str(audio_path))
        
        # Whisper encoder processes audio in 30-second blocks. Pad or trim.
        audio = whisper.pad_or_trim(audio)
        
        # Compute log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels).to(device)
        
        # Run encoder to get hidden states
        # Shape of mel: (n_mels, 3000) -> unsqueeze to add batch dim (1, n_mels, 3000)
        with torch.no_grad():
            # encoder output shape: (batch_size, sequence_len, hidden_dim)
            # For large-v3, sequence_len is 1500 (representing 30s), hidden_dim is 1280
            hidden_states = model.encoder(mel.unsqueeze(0))
            
            # Perform mean-pooling over the sequence length dimension (dim 1)
            # to get a single vector representing the phonetic signature
            mean_pooled = hidden_states.mean(dim=1).squeeze(0)
            
            # Move to CPU and convert to numpy
            embedding = mean_pooled.cpu().numpy()
            
        return embedding
    except Exception as e:
        logger.error(f"  Error extracting features for {audio_path.name}: {e}")
        return None


def run_extraction(model_name=None, device=None, use_enhanced=True):
    settings = load_settings()
    paths = get_data_paths(settings)
    
    # Configure parameters
    t_cfg = settings.get("transcription", {})
    model_name = model_name or t_cfg.get("model_name", "large-v3")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    project_root = paths["raw_audio"].parent.parent
    
    # Set up input directories (prefer enhanced audio if available)
    if use_enhanced:
        input_dir = project_root / "data" / "clean_segments"
        if not input_dir.exists() or not list(input_dir.glob("*.wav")):
            logger.warning("No files found in data/clean_segments. Falling back to data/processed.")
            input_dir = paths["processed"]
    else:
        input_dir = paths["processed"]
        
    # Output directory
    output_dir = project_root / "data" / "features" / "whisper_embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wav_files = sorted(list(input_dir.glob("*.wav")))
    if not wav_files:
        logger.error(f"No WAV files found in directory: {input_dir}")
        return
        
    logger.info(f"Loading Whisper model '{model_name}' on {device}...")
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.info("Retrying on CPU...")
        model = whisper.load_model(model_name, device="cpu")
        device = "cpu"
        
    logger.info(f"Extracting hidden states for {len(wav_files)} files...")
    
    embeddings = []
    file_names = []
    
    for fpath in tqdm(wav_files, desc="Extracting features"):
        out_path = output_dir / f"{fpath.stem}.npy"
        
        # Avoid redundant extraction if already present
        if out_path.exists():
            embedding = np.load(out_path)
            embeddings.append(embedding)
            file_names.append(fpath.name)
            continue
            
        embedding = extract_whisper_embedding(model, fpath, device)
        if embedding is not None:
            np.save(out_path, embedding)
            embeddings.append(embedding)
            file_names.append(fpath.name)
            
    if not embeddings:
        logger.error("No features were extracted successfully.")
        return
        
    # Compute the native speaker centroid
    embeddings_arr = np.array(embeddings)
    centroid = np.mean(embeddings_arr, axis=0)
    
    centroid_path = project_root / "data" / "features" / "urdu_native_centroid.npy"
    np.save(centroid_path, centroid)
    
    logger.info(f"🎉 Feature extraction complete!")
    logger.info(f"  - Segment embeddings saved to: {output_dir}")
    logger.info(f"  - Native speaker centroid saved to: {centroid_path}")
    logger.info(f"  - Feature dimension: {centroid.shape[0]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Whisper hidden states.")
    p.add_argument("--model", default=None, help="Whisper model size (default: large-v3).")
    p.add_argument("--device", default=None, help="Device to use (cuda/cpu).")
    p.add_argument("--raw", action="store_true", help="Use raw processed segments instead of DeepFilterNet enhanced.")
    args = p.parse_args()
    
    run_extraction(model_name=args.model, device=args.device, use_enhanced=not args.raw)
