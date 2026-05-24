#!/usr/bin/env python3
"""
Local‑only version of the Urdu‑Speech‑AI pipeline.
- Processes the audio in small micro‑batches (default 20 files per batch).
- Calls the original run_pipeline.py (which does noise reduction, silence splitting, etc.).
- Converts the resulting WAVs to 128 kbps MP3s and stores them in a permanent `final_processed/` folder.
- Works on Windows / macOS / Linux and falls back to CPU if no GPU is present.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# ----------------------------------------------------------------------
# 1️⃣ CONFIGURATION – UPDATE THESE PATHS TO MATCH YOUR LOCAL LAYOUT
# ----------------------------------------------------------------------
# Root folder of the project (where raw/, processed/, etc. live)
WORKING_DIR = Path(r"C:/Users/HP/OneDrive/Documents/Urdu-Speech-Ai")
# Folder containing the original raw audio files (the Kaggle "raw" directory)
AUDIO_ROOT = WORKING_DIR / "raw"
# Folder containing the original pipeline code (must have run_pipeline.py)
CODE_ROOT = WORKING_DIR / "src"

# ----------------------------------------------------------------------
# 2️⃣ INTERNAL DIRECTORIES (created automatically)
# ----------------------------------------------------------------------
RAW_DIR = WORKING_DIR / "raw_temp"
PROCESSED_DIR = WORKING_DIR / "processed_temp"
FINAL_PROCESSED = WORKING_DIR / "final_processed"

# ----------------------------------------------------------------------
# 3️⃣ CLEAN‑START – wipe any leftovers from a previous run
# ----------------------------------------------------------------------
print("Cleaning previous temporary data ...")
for p in (RAW_DIR, PROCESSED_DIR):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

# Remove stray zip files that belong to the older Kaggle version
for item in WORKING_DIR.glob("*.zip"):
    item.unlink()
for item in WORKING_DIR.glob("_processed_hashes.json"):
    item.unlink()

# ----------------------------------------------------------------------
# 4️⃣ GPU / CPU selection
# ----------------------------------------------------------------------
if shutil.which("nvidia-smi"):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print("CUDA GPUs detected – will use them if the pipeline supports CUDA.")
else:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("CUDA_DEVICE_ORDER", None)
    print("No GPU detected – falling back to CPU (will be slower).")

# ----------------------------------------------------------------------
# 5️⃣ VALIDATE INPUT PATHS
# ----------------------------------------------------------------------
if not AUDIO_ROOT.is_dir():
    raise FileNotFoundError(f"Audio root not found: {AUDIO_ROOT}")
if not CODE_ROOT.is_dir():
    raise FileNotFoundError(f"Code root not found: {CODE_ROOT}")

SCRIPT_FILE = CODE_ROOT / "run_pipeline.py"
if not SCRIPT_FILE.is_file():
    raise FileNotFoundError(f"run_pipeline.py missing from {CODE_ROOT}")

# ----------------------------------------------------------------------
# 6️⃣ GATHER ALL SOURCE AUDIO FILES
# ----------------------------------------------------------------------
all_source_files = [f for f in AUDIO_ROOT.rglob("*") if f.is_file() and not f.name.startswith('.')]
total_files = len(all_source_files)
print(f"\nAudio source      : {AUDIO_ROOT}")
print(f"Pipeline code    : {CODE_ROOT}")
print(f"{total_files:,} audio files discovered.\n")

# ----------------------------------------------------------------------
# 7️⃣ PREPARE FINAL DESTINATION
# ----------------------------------------------------------------------
FINAL_PROCESSED.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 8️⃣ MICRO‑BATCH PROCESSING LOOP
# ----------------------------------------------------------------------
MICRO_BATCH_SIZE = 20
total_batches = (total_files + MICRO_BATCH_SIZE - 1) // MICRO_BATCH_SIZE
print(f"Starting processing – {total_batches:,} batches (≈{MICRO_BATCH_SIZE} files per batch)")
global_start = time.time()

for start_idx in range(0, total_files, MICRO_BATCH_SIZE):
    batch_start = time.time()
    batch_files = all_source_files[start_idx:start_idx + MICRO_BATCH_SIZE]
    batch_no = (start_idx // MICRO_BATCH_SIZE) + 1

    print("\n" + "=" * 60)
    print(f"Batch {batch_no}/{total_batches} – {len(batch_files)} tracks")
    print("=" * 60)

    # Stage the current batch
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for src in batch_files:
        dst = RAW_DIR / src.name
        try:
            os.symlink(str(src), str(dst))
        except OSError:
            shutil.copy2(str(src), str(dst))

    # Execute the original pipeline (with make_archive disabled)
    python_cmd = (
        "import shutil, sys; "
        "shutil.make_archive = lambda *a, **k: print('make_archive bypassed'); "
        "sys.argv = ['run_pipeline.py', '--split-on-silence']; "
        f"exec(open(r'{SCRIPT_FILE.as_posix()}').read())"
    )
    subprocess.run([sys.executable, "-c", python_cmd], check=True)

    # Convert WAV → MP3 and move to final folder
    if PROCESSED_DIR.exists() and any(PROCESSED_DIR.iterdir()):
        wav_files = list(PROCESSED_DIR.rglob("*.wav"))
        if wav_files:
            print(f"Converting {len(wav_files)} WAV → MP3 (128 kbps)...")
        for wav_path in wav_files:
            mp3_path = wav_path.with_suffix('.mp3')
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            dest = FINAL_PROCESSED / mp3_path.name
            shutil.move(str(mp3_path), str(dest))
            wav_path.unlink()
    else:
        print("No processed output produced in this batch – skipping conversion.")

    # Clean temporary folders for next batch
    shutil.rmtree(RAW_DIR, ignore_errors=True)
    shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stray temp wav/mp3 files (Windows temp folder)
    temp_dir = Path(os.getenv('TMP', '/tmp'))
    for pat in ("*.wav", "*.mp3"):
        for p in temp_dir.rglob(pat):
            try:
                p.unlink()
            except Exception:
                pass

    batch_elapsed = time.time() - batch_start
    print(f"Batch {batch_no} done – {batch_elapsed:.2f}s elapsed")

# ----------------------------------------------------------------------
# 9️⃣ FINAL REPORT
# ----------------------------------------------------------------------
total_elapsed = time.time() - global_start
print("\n" + "=" * 70)
print("ALL BATCHES FINISHED")
print(f"Total runtime: {total_elapsed/60:.2f} minutes")
print(f"Processed MP3s are stored in: {FINAL_PROCESSED}")
print("=" * 70)
