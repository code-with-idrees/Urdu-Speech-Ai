"""
Audio Segmentation — Urdu Speech AI Pipeline (Step 3)

Slices DeepFilterNet-cleaned audio into short segments (5-15s) with
VAD-aware boundaries for accent probing.

Usage:
    python src/segment_audio.py
    python src/segment_audio.py --file "path/to/audio.wav"
    python src/segment_audio.py --duration 10
    python src/segment_audio.py --no-vad
"""

import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from utils import (
    load_settings, get_data_paths, load_manifest, save_manifest,
    make_segment_id, setup_logging, timestamp,
)

logger = setup_logging("segment_audio", log_file="logs/segment_audio.log")


def find_silence_boundary(audio, sr, target_sample, window_sec=2.0, thresh_db=-40.0):
    """Find nearest silence near target_sample to avoid mid-word cuts."""
    w = int(window_sec * sr)
    start = max(0, target_sample - w)
    end = min(len(audio), target_sample + w)
    seg = audio[start:end]
    if len(seg) == 0:
        return target_sample

    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    energy = []
    for i in range(0, len(seg) - frame_len, hop):
        rms = np.sqrt(np.mean(seg[i:i+frame_len] ** 2))
        energy.append((i + start, 20 * np.log10(rms + 1e-10)))

    silent = [(p, d) for p, d in energy if d < thresh_db]
    if silent:
        return min(silent, key=lambda x: abs(x[0] - target_sample))[0]
    return target_sample


def segment_file(audio_path, output_dir, poet_name, video_id, seg_dur=10,
                 overlap=2, min_dur=3, target_sr=16000, use_vad=True):
    """Segment a single audio file into chunks. Returns list of metadata dicts."""
    logger.info("  Loading: %s", audio_path.name)
    try:
        audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
    except Exception as e:
        logger.error("  Failed to load %s: %s", audio_path.name, e)
        return []

    total = len(audio)
    seg_samples = seg_dur * sr
    hop_samples = (seg_dur - overlap) * sr

    logger.info("  Duration: %.1f min | Segment: %ds | Overlap: %ds",
                (total / sr) / 60, seg_dur, overlap)

    results = []
    seg_num = 0
    pos = 0
    while pos < total:
        end = pos + seg_samples
        if use_vad and end < total:
            end = find_silence_boundary(audio, sr, end)

        chunk = audio[pos:min(end, total)]
        dur = len(chunk) / sr
        if dur < min_dur:
            break

        seg_id = make_segment_id(poet_name, video_id, seg_num)
        out_path = output_dir / f"{seg_id}.wav"
        sf.write(str(out_path), chunk, sr, subtype="PCM_16")

        results.append({
            "segment_id": seg_id, "source_file": audio_path.name,
            "poet": poet_name, "video_id": video_id, "segment_num": seg_num,
            "start_sec": pos / sr, "end_sec": min(end, total) / sr,
            "duration_sec": round(dur, 2), "sample_rate": sr,
            "output_file": f"{seg_id}.wav", "created_at": timestamp(),
        })
        seg_num += 1
        pos += hop_samples

    logger.info("  Created %d segments", seg_num)
    return results


def run_segmentation(single_file=None, use_vad=True, seg_duration=None, force=False):
    """Segment all clean audio files (or a single file)."""
    settings = load_settings()
    paths = get_data_paths(settings)
    cfg = settings.get("audio", {})

    # Use clean_raw (DeepFilterNet output) as the primary source
    project_root = paths["raw_audio"].parent.parent
    clean_raw_dir = project_root / "data" / "clean_raw"

    # Fallback to raw/raw if clean_raw doesn't exist yet
    if clean_raw_dir.exists() and any(clean_raw_dir.iterdir()):
        source_dir = clean_raw_dir
        logger.info("Using DeepFilterNet-cleaned audio from: %s", clean_raw_dir)
    else:
        source_dir = project_root / "raw" / "raw"
        if not source_dir.exists():
            source_dir = paths.get("raw_audio", project_root / "data" / "raw")
        logger.warning("clean_raw/ not found or empty, falling back to: %s", source_dir)

    seg_dir = paths["segments"]
    raw_manifest = load_manifest(source_dir / "manifest.json")
    seg_manifest_path = seg_dir / "manifest.json"
    seg_manifest = load_manifest(seg_manifest_path)

    # Override segment duration if provided via CLI
    default_dur = cfg.get("segment_duration_sec", 10)
    seg_dur = seg_duration if seg_duration is not None else default_dur
    overlap = cfg.get("segment_overlap_sec", 2)
    min_dur = cfg.get("min_segment_duration_sec", 3)
    target_sr = cfg.get("sample_rate", 16000)

    files = [Path(single_file)] if single_file else sorted(
        list(source_dir.rglob("*.mp3")) + list(source_dir.rglob("*.wav")) +
        list(source_dir.rglob("*.m4a")) + list(source_dir.rglob("*.flac"))
    )
    if not files:
        logger.warning("No audio files found in %s", source_dir)
        return

    logger.info("Found %d file(s) to segment (duration=%ds, overlap=%ds)",
                len(files), seg_dur, overlap)
    total_new = 0
    for idx, f in enumerate(files, 1):
        poet, vid = "unknown", f.stem[-11:]
        for v, m in raw_manifest.items():
            if isinstance(m, dict) and v in f.stem:
                poet, vid = m.get("poet", "unknown"), v
                break
        if f.name in seg_manifest.get("processed_files", []) and not force:
            if idx % 100 == 0:
                logger.info("⏭ Skipping already-processed files... (%d/%d)", idx, len(files))
            continue

        logger.info("[%d/%d] Segmenting: %s", idx, len(files), f.name)
        segs = segment_file(f, seg_dir, poet, vid, seg_dur, overlap,
                            min_dur, target_sr, use_vad)
        for s in segs:
            seg_manifest[s["segment_id"]] = s
        if f.name not in seg_manifest.setdefault("processed_files", []):
            seg_manifest["processed_files"].append(f.name)
        save_manifest(seg_manifest_path, seg_manifest)
        total_new += len(segs)

    logger.info("✅ Done: %d new segments created", total_new)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Segment audio into chunks for accent probing.")
    p.add_argument("--file", default=None, help="Single file to segment.")
    p.add_argument("--duration", type=int, default=None,
                   help="Segment duration in seconds (default: from settings.yaml, typically 10s).")
    p.add_argument("--no-vad", action="store_true", help="Disable VAD boundaries.")
    p.add_argument("--force", action="store_true", help="Force re-segmentation even if marked as processed.")
    args = p.parse_args()
    run_segmentation(single_file=args.file, use_vad=not args.no_vad,
                     seg_duration=args.duration, force=args.force)
