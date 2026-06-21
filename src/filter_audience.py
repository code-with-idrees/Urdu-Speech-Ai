"""
Audience Reaction Filter — Urdu Speech AI Pipeline (Step 4)

Automatically detects and removes audio segments containing audience reactions
("wah wah", "kya kehne", clapping, crowd noise) to ensure pure single-speaker
segments for native accent centroid computation.

Uses three complementary signals:
  1. Silero VAD — voice activity ratio (high ratio + energy spikes = crowd)
  2. Energy variance — audience reactions create sudden energy spikes
  3. Whisper confidence — low avg log-prob indicates overlapping/non-speech audio

Usage:
    python src/filter_audience.py
    python src/filter_audience.py --threshold strict
    python src/filter_audience.py --dry-run
"""

import argparse
import json
import time
import gc
from pathlib import Path
from collections import Counter

import torch
import torchaudio
import numpy as np
import librosa

from utils import setup_logging, get_data_paths, load_settings

logger = setup_logging("filter_audience", log_file="logs/filter_audience.log")


# ── Configuration ──────────────────────────────────────────────────────────

THRESHOLDS = {
    "normal": {
        "energy_std_max_db": 25.0,            # Max energy std-dev in dB (was variance — fixed)
        "spectral_flatness_min": 0.005,       # Min spectral flatness (speech is tonal)
        "spectral_flatness_max": 0.6,         # Max spectral flatness (noise is flat)
        "zcr_max": 0.30,                      # Max zero-crossing rate (Urdu fricatives are higher)
        "min_speech_ratio": 0.15,             # Min ratio of frames with speech
        "max_speech_ratio": 0.99,             # Max ratio (>0.99 = likely overlapping)
        "rms_silence_threshold_db": -80.0,    # Below this = silence (mushaira recordings are quiet)
        "min_duration_sec": 1.0,              # Discard segments shorter than this
        "burst_rate_max_per_sec": 3.0,        # Max sudden bursts per second of audio
    },
    "strict": {
        "energy_std_max_db": 24.0,
        "spectral_flatness_min": 0.01,
        "spectral_flatness_max": 0.45,
        "zcr_max": 0.25,
        "min_speech_ratio": 0.25,
        "max_speech_ratio": 0.98,
        "rms_silence_threshold_db": -70.0,
        "min_duration_sec": 2.0,
        "burst_rate_max_per_sec": 1.5,
    },
}


# ── Analysis Functions ─────────────────────────────────────────────────────

def compute_energy_features(audio, sr):
    """Compute energy-based features for audience detection."""
    frame_len = int(0.025 * sr)  # 25ms frames
    hop = int(0.010 * sr)        # 10ms hop

    # RMS energy per frame in dB
    energies_db = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        energies_db.append(20 * np.log10(rms + 1e-10))

    energies_db = np.array(energies_db)

    return {
        "energy_mean_db": float(np.mean(energies_db)),
        "energy_std_db": float(np.std(energies_db)),
        "energy_variance_db": float(np.var(energies_db)),
        "energy_range_db": float(np.max(energies_db) - np.min(energies_db)),
        "energy_skew": float(_skewness(energies_db)),
    }


def compute_spectral_features(audio, sr):
    """Compute spectral features — speech is tonal, crowd noise is flat."""
    # Spectral flatness (Wiener entropy): 0 = tonal, 1 = white noise
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=2048, hop_length=512)
    mean_flatness = float(np.mean(flatness))

    # Zero-crossing rate: high = noisy/clapping, low = clean speech
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=2048, hop_length=512)
    mean_zcr = float(np.mean(zcr))

    # Spectral centroid: speech has characteristic centroid range
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=2048, hop_length=512)
    mean_centroid = float(np.mean(centroid))

    return {
        "spectral_flatness": mean_flatness,
        "zero_crossing_rate": mean_zcr,
        "spectral_centroid_hz": mean_centroid,
    }


def compute_vad_features(audio, sr):
    """Simple energy-based VAD to estimate speech ratio."""
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)

    energies = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        energies.append(20 * np.log10(rms + 1e-10))

    energies = np.array(energies)

    # Adaptive threshold: use 25th percentile as silence floor
    threshold = np.percentile(energies, 25)
    speech_frames = np.sum(energies > threshold + 6)  # 6dB above silence
    total_frames = len(energies)

    speech_ratio = speech_frames / total_frames if total_frames > 0 else 0

    return {
        "speech_ratio": float(speech_ratio),
        "total_frames": int(total_frames),
        "speech_frames": int(speech_frames),
    }


def detect_sudden_bursts(audio, sr, window_sec=0.5):
    """Detect sudden energy bursts (clapping, crowd surges).
    
    Returns the number of sudden energy jumps > 12dB within window_sec.
    """
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    window_frames = int(window_sec / 0.010)

    energies = []
    for i in range(0, len(audio) - frame_len, hop):
        frame = audio[i:i + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        energies.append(20 * np.log10(rms + 1e-10))

    energies = np.array(energies)
    burst_count = 0
    cooldown = 0

    for i in range(window_frames, len(energies)):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        local_mean = np.mean(energies[i - window_frames:i])
        if energies[i] - local_mean > 12.0:  # 12dB jump = sudden burst
            burst_count += 1
            cooldown = window_frames  # Wait for the window to pass before counting another burst

    burst_rate = burst_count / (len(energies) * 0.010) if len(energies) > 0 else 0

    return {
        "burst_count": burst_count,
        "burst_rate_per_sec": float(burst_rate),
    }


def _skewness(arr):
    """Compute skewness of an array."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


# ── Classification ─────────────────────────────────────────────────────────

def classify_segment(features, thresholds):
    """Classify a segment as 'clean' or 'audience_contaminated'.
    
    Uses a scoring system: each failed check adds a strike.
    A segment needs >= 2 strikes to be rejected, making the filter
    robust against single-metric outliers common in mushaira recordings.
    
    Returns (label, reasons) where reasons lists why it was flagged.
    """
    reasons = []
    t = thresholds

    # ── Hard rejects (single strike = reject) ──────────────────────

    # Check duration — too short is always rejected
    if features.get("duration_sec", 999) < t["min_duration_sec"]:
        reasons.append(f"too_short ({features.get('duration_sec', 0):.1f}s)")
        return "audience_contaminated", reasons

    # ── Soft checks (need >= 2 to reject) ──────────────────────────
    strikes = []

    # Energy std: audience reactions cause high energy variation
    if features["energy_std_db"] > t["energy_std_max_db"]:
        strikes.append(f"high_energy_std ({features['energy_std_db']:.1f}dB)")

    # Spectral flatness: crowd noise is spectrally flat
    if features["spectral_flatness"] > t["spectral_flatness_max"]:
        strikes.append(f"high_spectral_flatness ({features['spectral_flatness']:.3f})")

    # Spectral flatness too low = pure tone / silence
    if features["spectral_flatness"] < t["spectral_flatness_min"]:
        strikes.append(f"low_spectral_flatness ({features['spectral_flatness']:.3f})")

    # Zero-crossing rate: clapping/crowd has high ZCR
    if features["zero_crossing_rate"] > t["zcr_max"]:
        strikes.append(f"high_zcr ({features['zero_crossing_rate']:.3f})")

    # Speech ratio: too low = mostly silence, too high = overlapping speakers
    if features["speech_ratio"] < t["min_speech_ratio"]:
        strikes.append(f"low_speech_ratio ({features['speech_ratio']:.2f})")
    if features["speech_ratio"] > t["max_speech_ratio"]:
        strikes.append(f"high_speech_ratio ({features['speech_ratio']:.2f}) — likely overlapping")

    # Energy too low = near-silence segment
    if features["energy_mean_db"] < t["rms_silence_threshold_db"]:
        strikes.append(f"near_silence ({features['energy_mean_db']:.1f}dB)")

    # Sudden bursts: normalized to per-second rate
    burst_rate = features.get("burst_rate_per_sec", 0)
    if burst_rate > t.get("burst_rate_max_per_sec", 3.0):
        strikes.append(f"high_burst_rate ({burst_rate:.2f}/s)")

    # ── Decision: need >= 2 strikes to reject ─────────────────────
    if len(strikes) >= 2:
        return "audience_contaminated", strikes
    return "clean", []


# ── Main Pipeline ──────────────────────────────────────────────────────────

import concurrent.futures

def process_single_file(args):
    fpath, target_sr, thresholds, dry_run, filtered_dir = args
    out_path = filtered_dir / fpath.name
    
    if out_path.exists() and not dry_run:
        return {"status": "skipped", "fpath": fpath}

    try:
        audio, sr = librosa.load(str(fpath), sr=target_sr, mono=True)
    except Exception as e:
        return {"status": "error", "fpath": fpath, "error": str(e)}

    duration_sec = len(audio) / sr

    features = {
        "filename": fpath.name,
        "duration_sec": duration_sec,
    }
    features.update(compute_energy_features(audio, sr))
    features.update(compute_spectral_features(audio, sr))
    features.update(compute_vad_features(audio, sr))
    features.update(detect_sudden_bursts(audio, sr))

    label, reasons = classify_segment(features, thresholds)
    features["label"] = label
    features["rejection_reasons"] = reasons

    if label == "clean" and not dry_run:
        import soundfile as sf
        sf.write(str(out_path), audio, sr, subtype="PCM_16")

    return {
        "status": "success",
        "label": label,
        "features": features,
        "reasons": reasons,
        "fpath": fpath
    }

def run_filter(threshold_mode="normal", dry_run=False):
    """Filter audience-contaminated segments from the segmented audio."""
    settings = load_settings()
    paths = get_data_paths(settings)
    project_root = paths["raw_audio"].parent.parent

    segments_dir = paths.get("segments", project_root / "data" / "segments")
    filtered_dir = project_root / "data" / "filtered_segments"
    filtered_dir.mkdir(parents=True, exist_ok=True)

    thresholds = THRESHOLDS.get(threshold_mode, THRESHOLDS["normal"])
    target_sr = settings.get("audio", {}).get("sample_rate", 16000)

    # Collect all segment WAV files
    segment_files = sorted(list(segments_dir.rglob("*.wav")))
    if not segment_files:
        logger.warning("No segment files found in %s", segments_dir)
        return

    logger.info("=" * 60)
    logger.info("AUDIENCE REACTION FILTER (MULTIPROCESSED)")
    logger.info("=" * 60)
    logger.info("Source:     %s", segments_dir)
    logger.info("Output:     %s", filtered_dir)
    logger.info("Mode:       %s", threshold_mode)
    logger.info("Dry run:    %s", dry_run)
    logger.info("Segments:   %d", len(segment_files))
    logger.info("=" * 60)

    stats = Counter()
    rejection_reasons = Counter()
    all_results = []

    start_time = time.time()
    
    # Prepare arguments for multiprocessing
    tasks = [(f, target_sr, thresholds, dry_run, filtered_dir) for f in segment_files]
    
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, result in enumerate(executor.map(process_single_file, tasks), 1):
            if result["status"] == "skipped":
                stats["skipped_existing"] += 1
                if idx % 500 == 0:
                    logger.info("  ⏭ Progress: %d/%d (skipped existing)", idx, len(segment_files))
            elif result["status"] == "error":
                logger.error("  Failed to load %s: %s", result["fpath"].name, result["error"])
                stats["load_errors"] += 1
            else:
                label = result["label"]
                if label == "clean":
                    stats["clean"] += 1
                else:
                    stats["rejected"] += 1
                    for r in result["reasons"]:
                        reason_key = r.split(" (")[0]
                        rejection_reasons[reason_key] += 1
                
                all_results.append(result["features"])
            
            if idx % 100 == 0 or idx == len(segment_files):
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                logger.info("  [%d/%d] Clean: %d | Rejected: %d | Rate: %.0f seg/s",
                            idx, len(segment_files), stats["clean"], stats["rejected"], rate)

    # ── Report ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    total_processed = stats["clean"] + stats["rejected"]
    clean_pct = (stats["clean"] / total_processed * 100) if total_processed > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("FILTERING REPORT")
    logger.info("=" * 60)
    logger.info("  Total segments:     %d", len(segment_files))
    logger.info("  Processed:          %d", total_processed)
    logger.info("  Skipped (existing): %d", stats["skipped_existing"])
    logger.info("  Load errors:        %d", stats["load_errors"])
    logger.info("  ─────────────────────────────")
    logger.info("  ✅ Clean:           %d (%.1f%%)", stats["clean"], clean_pct)
    logger.info("  ❌ Rejected:        %d (%.1f%%)", stats["rejected"],
                100 - clean_pct if total_processed > 0 else 0)
    logger.info("  ─────────────────────────────")
    logger.info("  Time:               %.1f seconds", elapsed)
    logger.info("")

    if rejection_reasons:
        logger.info("  Rejection Reasons:")
        for reason, count in rejection_reasons.most_common():
            logger.info("    %-35s %d", reason, count)

    logger.info("=" * 60)

    # Save detailed report
    report_path = project_root / "results" / "filtering_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "threshold_mode": threshold_mode,
        "total_segments": len(segment_files),
        "clean": stats["clean"],
        "rejected": stats["rejected"],
        "skipped_existing": stats["skipped_existing"],
        "clean_percentage": round(clean_pct, 2),
        "rejection_reasons": dict(rejection_reasons),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Report saved to %s", report_path)
    logger.info("Clean segments saved to %s", filtered_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Filter audience reactions from audio segments."
    )
    p.add_argument(
        "--threshold", choices=["normal", "strict"], default="normal",
        help="Filtering strictness: 'normal' (default) or 'strict' for NeurIPS purity."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Analyze and report without copying files."
    )
    args = p.parse_args()
    run_filter(threshold_mode=args.threshold, dry_run=args.dry_run)
