"""
Audio Preprocessing — Urdu Speech AI Pipeline (Step 3)

Normalizes segmented audio: resamples to 16kHz mono WAV, normalizes
loudness, and trims leading/trailing silence.

Usage:
    python src/preprocess_audio.py
"""

import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from utils import load_settings, get_data_paths, setup_logging, timestamp

logger = setup_logging("preprocess", log_file="logs/preprocess.log")


def normalize_audio(audio, target_db=-20.0):
    """Normalize audio to a target dB level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    current_db = 20 * np.log10(rms)
    gain = 10 ** ((target_db - current_db) / 20)
    return np.clip(audio * gain, -1.0, 1.0)


def preprocess_file(input_path, output_path, target_sr=16000):
    """Load, normalize, trim silence, and save as 16kHz mono WAV."""
    try:
        audio, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
    except Exception as e:
        logger.error("  Failed to load %s: %s", input_path.name, e)
        return False

    # Trim leading/trailing silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)

    # Normalize loudness
    audio_norm = normalize_audio(audio_trimmed)

    sf.write(str(output_path), audio_norm, target_sr, subtype="PCM_16")
    dur = len(audio_norm) / target_sr
    logger.info("  ✅ %s → %.1fs", input_path.name, dur)
    return True


def run_preprocessing():
    """Preprocess all segments in data/segments/ → data/processed/."""
    settings = load_settings()
    paths = get_data_paths(settings)
    target_sr = settings.get("audio", {}).get("sample_rate", 16000)

    seg_dir = paths["segments"]
    proc_dir = paths["processed"]

    wav_files = sorted(seg_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No WAV files in %s", seg_dir)
        return

    logger.info("Preprocessing %d segment(s)", len(wav_files))
    ok, fail = 0, 0
    for f in wav_files:
        out = proc_dir / f.name
        if out.exists():
            logger.info("  ⏭ %s already processed", f.name)
            continue
        if preprocess_file(f, out, target_sr):
            ok += 1
        else:
            fail += 1

    logger.info("Done: %d processed, %d failed", ok, fail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and clean audio segments.")
    parser.parse_args()
    run_preprocessing()
