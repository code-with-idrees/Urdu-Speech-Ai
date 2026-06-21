"""
Accent Fidelity Benchmark Engine — Urdu Speech AI Pipeline

Evaluates synthesized TTS audio against native speaker centroids by:
  1. Extracting Whisper hidden-state embeddings for the TTS audio files.
  2. Computing Cosine Distance from the native centroid.
  3. Computing the Fréchet Audio Distance (FAD) between the TTS system's
     embedding distribution and the native speaker distribution.

Usage:
    python src/accent_benchmark.py --tts_dir data/tts_samples/model_a/ --lang ur
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
import torch
import whisper
from tqdm import tqdm

from utils import setup_logging, get_data_paths, load_settings

logger = setup_logging("accent_benchmark", log_file="logs/accent_benchmark.log")


def compute_frechet_distance(mu_native, cov_native, mu_tts, cov_tts):
    """Compute the Fréchet Distance between two multivariate Gaussians.
    
    Falls back to a diagonal variance approximation if scipy is not installed
    or if the covariance matrix calculation is numerically unstable.
    """
    try:
        from scipy.linalg import sqrtm
        # Calculate sum squared difference between means
        diff = mu_native - mu_tts
        mean_diff_sq = np.sum(diff ** 2)
        
        # Calculate matrix square root of product of covariances
        covmean = sqrtm(cov_native.dot(cov_tts))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fd = mean_diff_sq + np.trace(cov_native + cov_tts - 2.0 * covmean)
        return float(fd)
    except Exception as e:
        # Diagonal fallback (highly stable, only relies on numpy)
        var_native = np.diag(cov_native) if cov_native.ndim > 1 else cov_native
        var_tts = np.diag(cov_tts) if cov_tts.ndim > 1 else cov_tts
        
        diff = mu_native - mu_tts
        mean_diff_sq = np.sum(diff ** 2)
        var_diff_sq = np.sum((np.sqrt(np.maximum(var_native, 1e-10)) - np.sqrt(np.maximum(var_tts, 1e-10))) ** 2)
        return float(mean_diff_sq + var_diff_sq)


def extract_embeddings_for_dir(model, audio_dir, device):
    """Run Whisper encoder on all WAV files in a directory to extract embeddings."""
    audio_paths = sorted(list(audio_dir.glob("*.wav")))
    if not audio_paths:
        logger.error(f"No WAV files found in: {audio_dir}")
        return None, []
        
    logger.info(f"Extracting embeddings for {len(audio_paths)} TTS files...")
    embeddings = []
    file_names = []
    
    for fpath in tqdm(audio_paths, desc="Processing TTS files"):
        try:
            audio = whisper.load_audio(str(fpath))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels).to(device)
            
            with torch.no_grad():
                hidden_states = model.encoder(mel.unsqueeze(0))
                mean_pooled = hidden_states.mean(dim=1).squeeze(0)
                embeddings.append(mean_pooled.cpu().numpy())
                file_names.append(fpath.name)
        except Exception as e:
            logger.error(f"  Error processing {fpath.name}: {e}")
            
    if not embeddings:
        return None, []
        
    return np.array(embeddings), file_names


def run_benchmark(tts_dir_path, lang="ur", model_name=None, device=None):
    settings = load_settings()
    paths = get_data_paths(settings)
    project_root = paths["raw_audio"].parent.parent
    
    # Paths for centroids and native embeddings
    features_dir = project_root / "data" / "features"
    centroid_path = features_dir / f"{lang}_native_centroid.npy"
    
    if not centroid_path.exists():
        logger.error(f"Native centroid file not found: {centroid_path}")
        logger.error("Please run extract_whisper_features.py on your native dataset first.")
        sys.exit(1)
        
    logger.info(f"Loading native {lang.upper()} centroid and reference features...")
    native_centroid = np.load(centroid_path)
    
    # Try to load all individual native embeddings to calculate covariance matrix
    native_embeddings_dir = features_dir / "whisper_embeddings"
    native_emb_files = list(native_embeddings_dir.glob("*.npy"))
    if len(native_emb_files) > 1:
        native_embs = np.array([np.load(f) for f in native_emb_files])
        native_cov = np.cov(native_embs, rowvar=False)
    else:
        # Fallback if only the centroid is present
        native_cov = np.eye(len(native_centroid)) * 0.1
        logger.warning("Could not calculate actual native covariance matrix (need individual embeddings). Using identity placeholder.")

    # 1. Setup Whisper model
    t_cfg = settings.get("transcription", {})
    model_name = model_name or t_cfg.get("model_name", "large-v3")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading Whisper model '{model_name}' on {device}...")
    model = whisper.load_model(model_name, device=device)
    
    # 2. Extract TTS embeddings
    tts_dir = Path(tts_dir_path)
    tts_embs, file_names = extract_embeddings_for_dir(model, tts_dir, device)
    if tts_embs is None:
        logger.error("Failed to extract any TTS embeddings.")
        sys.exit(1)
        
    # 3. Calculate metrics
    # A. Cosine Similarities & Distances
    # Normalize vectors for cosine calculations
    native_norm = native_centroid / np.linalg.norm(native_centroid)
    tts_norms = tts_embs / np.linalg.norm(tts_embs, axis=1, keepdims=True)
    
    cosine_similarities = np.dot(tts_norms, native_norm)
    cosine_distances = 1.0 - cosine_similarities
    
    # B. TTS Distribution parameters
    tts_mean = np.mean(tts_embs, axis=0)
    tts_cov = np.cov(tts_embs, rowvar=False) if len(tts_embs) > 1 else np.eye(len(tts_mean)) * 0.1
    
    # C. Fréchet Audio Distance
    fad_score = compute_frechet_distance(native_centroid, native_cov, tts_mean, tts_cov)
    
    # 4. Generate report
    avg_cosine_dist = float(np.mean(cosine_distances))
    avg_cosine_sim = float(np.mean(cosine_similarities))
    
    # Normalize FAD to an intuitive 0.0 - 1.0 "Fidelity Score" where 1.0 is identical
    # Typically, FAD ranges from 0 (identical) to 50+ (very different)
    fidelity_score = float(np.exp(-fad_score / 10.0))  # Exponential decay scaling
    
    results = {
        "language": lang,
        "tts_directory": str(tts_dir),
        "num_evaluated_files": len(file_names),
        "metrics": {
            "average_cosine_similarity": avg_cosine_sim,
            "average_cosine_distance": avg_cosine_dist,
            "frechet_audio_distance": fad_score,
            "accent_fidelity_score": fidelity_score,
        },
        "per_file": [
            {
                "filename": filename,
                "cosine_similarity": float(sim),
                "cosine_distance": float(dist),
            }
            for filename, sim, dist in zip(file_names, cosine_similarities, cosine_distances)
        ]
    }
    
    # Save Report
    reports_dir = project_root / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"tts_accent_report_{tts_dir.name}.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    logger.info("=" * 60)
    logger.info(f"BENCHMARK RESULTS FOR {tts_dir.name.upper()} ({lang.upper()})")
    logger.info("=" * 60)
    logger.info(f"  Files Evaluated:          {len(file_names)}")
    logger.info(f"  Average Cosine Similarity: {avg_cosine_sim:.4f}")
    logger.info(f"  Average Cosine Distance:   {avg_cosine_dist:.4f}")
    logger.info(f"  Fréchet Audio Distance:    {fad_score:.4f}")
    logger.info(f"  ------------------------------------------------")
    bar = "█" * int(fidelity_score * 30)
    logger.info(f"  Accent Fidelity Score:     {bar} {fidelity_score:.4f} (1.0 = native-like)")
    logger.info("=" * 60)
    logger.info(f"Detailed JSON report saved to: {report_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate TTS accent fidelity against native centroids.")
    p.add_argument("--tts_dir", required=True, help="Directory containing synthesized TTS WAV files.")
    p.add_argument("--lang", default="ur", help="Language code (default: ur).")
    p.add_argument("--model", default=None, help="Whisper model size.")
    p.add_argument("--device", default=None, help="Device to use.")
    args = p.parse_args()
    
    run_benchmark(tts_dir_path=args.tts_dir, lang=args.lang, model_name=args.model, device=args.device)
