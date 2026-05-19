"""
Audio Segmentation — Urdu Speech AI Pipeline (Step 2)

Slices raw audio into fixed-duration segments with VAD-aware boundaries.

Usage:
    python src/segment_audio.py
    python src/segment_audio.py --file "path/to/audio.mp3"
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


def segment_file(audio_path, output_dir, poet_name, video_id, seg_dur=60,
                 overlap=5, min_dur=30, target_sr=16000, use_vad=True):
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

    logger.info("  Duration: %.1f min | Segment: %ds", (total / sr) / 60, seg_dur)

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


def run_segmentation(single_file=None, use_vad=True):
    """Segment all raw audio files (or a single file)."""
    settings = load_settings()
    paths = get_data_paths(settings)
    cfg = settings.get("audio", {})
    raw_dir, seg_dir = paths["raw_audio"], paths["segments"]

    raw_manifest = load_manifest(raw_dir / "manifest.json")
    seg_manifest_path = seg_dir / "manifest.json"
    seg_manifest = load_manifest(seg_manifest_path)

    files = [Path(single_file)] if single_file else sorted(
        list(raw_dir.glob("*.mp3")) + list(raw_dir.glob("*.wav")))
    if not files:
        logger.warning("No audio files found in %s", raw_dir)
        return

    logger.info("Found %d file(s) to segment", len(files))
    total_new = 0
    for f in files:
        poet, vid = "unknown", f.stem[-11:]
        for v, m in raw_manifest.items():
            if v in f.stem:
                poet, vid = m.get("poet", "unknown"), v
                break
        if f.name in seg_manifest.get("processed_files", []):
            logger.info("⏭ Skipping: %s", f.name)
            continue

        segs = segment_file(f, seg_dir, poet, vid,
                            cfg.get("segment_duration_sec", 60),
                            cfg.get("segment_overlap_sec", 5),
                            cfg.get("min_segment_duration_sec", 30),
                            cfg.get("sample_rate", 16000), use_vad)
        for s in segs:
            seg_manifest[s["segment_id"]] = s
        seg_manifest.setdefault("processed_files", []).append(f.name)
        save_manifest(seg_manifest_path, seg_manifest)
        total_new += len(segs)

    logger.info("Done: %d new segments", total_new)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Segment raw audio into chunks.")
    p.add_argument("--file", default=None, help="Single file to segment.")
    p.add_argument("--no-vad", action="store_true", help="Disable VAD boundaries.")
    args = p.parse_args()
    run_segmentation(single_file=args.file, use_vad=not args.no_vad)
