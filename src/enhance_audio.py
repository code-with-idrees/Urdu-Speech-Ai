"""
Audio Enhancement — Urdu Speech AI Pipeline

Uses DeepFilterNet (df.enhance Python API) to remove background noise while
preserving the poetry vocals. This runs natively on GPU (if available) and
outputs cleaned versions of both the raw audios and segments (cuttings).

Handles arbitrarily long files (even 22+ hours) by streaming chunks from disk
instead of loading the entire file into RAM.

Usage:
    python src/enhance_audio.py
"""

import sys
import time
import gc
from pathlib import Path

import torch
import torchaudio
from df.enhance import enhance, init_df, load_audio, save_audio
from utils import setup_logging, get_data_paths

logger = setup_logging("enhance_audio", log_file="logs/enhance_audio.log")

# ── Chunk size config ──────────────────────────────────────────────────────
CHUNK_SECONDS = 30          # Seconds per GPU chunk (keeps VRAM usage low)
STREAM_SECONDS = 300        # 5-min disk-read chunks for very long files
RAM_THRESHOLD_HOURS = 1.0   # Files longer than this use disk-streaming


def enhance_audio_chunked(model, df_state, audio_tensor):
    """Enhance a loaded audio tensor in GPU-safe chunks.
    
    For audio already loaded into RAM — splits into 30-second pieces,
    runs each through DeepFilterNet, then stitches back together.
    """
    sr = df_state.sr()
    chunk_samples = int(CHUNK_SECONDS * sr)
    total_chunks = (audio_tensor.shape[1] + chunk_samples - 1) // chunk_samples

    if total_chunks <= 1:
        # Short enough to process in one shot
        with torch.no_grad():
            return enhance(model, df_state, audio_tensor.contiguous())

    enhanced_chunks = []
    with torch.no_grad():
        for ci, i in enumerate(range(0, audio_tensor.shape[1], chunk_samples), 1):
            chunk = audio_tensor[:, i:i + chunk_samples].contiguous()
            enhanced_chunk = enhance(model, df_state, chunk)
            enhanced_chunks.append(enhanced_chunk.cpu())
            torch.cuda.empty_cache()
            if ci % 20 == 0 or ci == total_chunks:
                logger.info(f"      Chunk {ci}/{total_chunks}")

    return torch.cat(enhanced_chunks, dim=1)


def enhance_file_streaming(model, df_state, input_path, output_path):
    """Enhance a very long audio file by streaming from disk.
    
    Reads STREAM_SECONDS at a time from the file, enhances each
    stream-chunk in CHUNK_SECONDS GPU sub-chunks, and writes results
    incrementally. This keeps RAM usage constant regardless of file length.
    """
    sr = df_state.sr()

    # Get file info without loading
    info = torchaudio.info(str(input_path))
    file_sr = info.sample_rate
    total_frames = info.num_frames
    duration_s = total_frames / file_sr
    logger.info(f"    Streaming mode — Duration: {duration_s/3600:.1f} hrs, SR: {file_sr}")

    stream_frames = int(STREAM_SECONDS * file_sr)
    all_enhanced = []
    stream_idx = 0
    total_streams = (total_frames + stream_frames - 1) // stream_frames

    for offset in range(0, total_frames, stream_frames):
        stream_idx += 1
        num_frames = min(stream_frames, total_frames - offset)

        # Read a 5-minute chunk from disk
        waveform, orig_sr = torchaudio.load(
            str(input_path), frame_offset=offset, num_frames=num_frames
        )

        # Resample to DeepFilterNet sample rate if needed
        if orig_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_sr, sr)
            waveform = resampler(waveform)

        # Make mono if stereo (DeepFilterNet expects mono or handles it)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        logger.info(f"    Stream block {stream_idx}/{total_streams} "
                     f"({offset/file_sr/60:.0f}–{(offset+num_frames)/file_sr/60:.0f} min)")

        # Enhance this 5-min block in 30-second GPU chunks
        enhanced_block = enhance_audio_chunked(model, df_state, waveform)
        all_enhanced.append(enhanced_block.cpu())

        # Aggressively free memory
        del waveform, enhanced_block
        torch.cuda.empty_cache()
        gc.collect()

    # Concatenate and save
    final_audio = torch.cat(all_enhanced, dim=1)
    save_audio(str(output_path), final_audio, sr)
    del final_audio, all_enhanced
    gc.collect()


def process_file(model, df_state, input_path, output_path):
    """Decide whether to use in-memory or streaming enhancement."""
    # Get duration without loading
    try:
        info = torchaudio.info(str(input_path))
        duration_hours = info.num_frames / info.sample_rate / 3600
    except Exception:
        duration_hours = 0  # Fallback: try loading normally

    if duration_hours > RAM_THRESHOLD_HOURS:
        # Very long file → stream from disk
        logger.info(f"    Large file ({duration_hours:.1f}h) → using disk streaming")
        enhance_file_streaming(model, df_state, input_path, output_path)
    else:
        # Normal file → load into RAM, chunk for GPU
        audio, _ = load_audio(str(input_path), sr=df_state.sr())
        duration_s = audio.shape[1] / df_state.sr()
        logger.info(f"    Audio duration: {duration_s/60:.1f} min")
        enhanced = enhance_audio_chunked(model, df_state, audio)
        save_audio(str(output_path), enhanced, df_state.sr())
        del audio, enhanced
        torch.cuda.empty_cache()
        gc.collect()


def run_enhancement():
    paths = get_data_paths()
    project_root = paths["raw_audio"].parent.parent

    # Handle the raw directory path (User specified D:\Urdu-Speech-AI\raw\raw)
    raw_dir = project_root / "raw" / "raw"
    if not raw_dir.exists():
        raw_dir = paths.get("raw_audio", project_root / "data" / "raw")

    segments_dir = paths.get("segments", project_root / "data" / "segments")

    # Create safe output directories to avoid destructive overwriting
    clean_raw_dir = project_root / "data" / "clean_raw"
    clean_segments_dir = project_root / "data" / "clean_segments"

    clean_raw_dir.mkdir(parents=True, exist_ok=True)
    clean_segments_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing DeepFilterNet model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Monkey-patch DeepFilterNet to prevent it from crashing if 'git' is not installed
        import df.utils
        import df.logger as df_logger_mod
        df.utils.get_commit_hash = lambda: "unknown"
        df.utils.get_git_root = lambda: ""
        df.utils.get_branch_name = lambda: "unknown"
        if hasattr(df_logger_mod, 'get_commit_hash'):
            df_logger_mod.get_commit_hash = lambda: "unknown"
        if hasattr(df_logger_mod, 'get_branch_name'):
            df_logger_mod.get_branch_name = lambda: "unknown"

        model, df_state, _ = init_df()
        logger.info(f"Model loaded successfully on {device}.")
    except Exception as e:
        import traceback
        logger.error(f"Failed to initialize DeepFilterNet: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    # ── 1. Enhance full raw files ──────────────────────────────────────────
    raw_files = sorted(
        list(raw_dir.rglob("*.mp3")) + list(raw_dir.rglob("*.wav")) +
        list(raw_dir.rglob("*.m4a")) + list(raw_dir.rglob("*.ogg")) +
        list(raw_dir.rglob("*.flac"))
    )
    if raw_files:
        logger.info(f"Found {len(raw_files)} raw audio files to enhance.")
        for idx, fpath in enumerate(raw_files, 1):
            out_path = clean_raw_dir / f"{fpath.stem}_clean.wav"
            if out_path.exists():
                logger.info(f"  ⏭ [{idx}/{len(raw_files)}] Skipping {fpath.name}, already enhanced.")
                continue

            logger.info(f"  [{idx}/{len(raw_files)}] Enhancing raw audio: {fpath.name}")
            try:
                start_time = time.time()
                process_file(model, df_state, fpath, out_path)
                elapsed = time.time() - start_time
                logger.info(f"    ✅ Done in {elapsed:.1f}s: {fpath.name}")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(f"    ❌ Error enhancing {fpath.name}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
    else:
        logger.warning(f"No raw audio files found in {raw_dir}")

    # ── 2. Enhance segment files (cuttings) ────────────────────────────────
    segment_files = sorted(list(segments_dir.rglob("*.wav")))
    if segment_files:
        logger.info(f"Found {len(segment_files)} segment files to enhance.")
        for idx, fpath in enumerate(segment_files, 1):
            out_path = clean_segments_dir / fpath.name
            if out_path.exists():
                if idx % 100 == 0:
                    logger.info(f"  ⏭ Skipping segments... ({idx}/{len(segment_files)} checked)")
                continue

            if idx % 10 == 0 or idx == 1:
                logger.info(f"  [{idx}/{len(segment_files)}] Enhancing segment: {fpath.name}")

            try:
                process_file(model, df_state, fpath, out_path)
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(f"  ❌ Error enhancing {fpath.name}: {e}")
                torch.cuda.empty_cache()
                gc.collect()
    else:
        logger.warning(f"No segment files found in {segments_dir}")

    logger.info("🎉 All enhancement tasks completed! Results in data/clean_raw and data/clean_segments.")


if __name__ == "__main__":
    try:
        import df
    except ImportError:
        print("Error: deepfilternet python package is not installed. Please run `pip install deepfilternet`.")
        sys.exit(1)

    run_enhancement()
