"""
Urdu ASR Transcription — Urdu Speech AI Pipeline (Step 4)

Transcribes processed audio segments using OpenAI Whisper.
Saves transcripts as JSON sidecar files alongside annotations.

Usage:
    python src/transcribe.py
    python src/transcribe.py --model medium   # use smaller model
    python src/transcribe.py --device cpu     # force CPU
"""

import argparse
import json
import os
from pathlib import Path

# Add current directory to PATH so Whisper can find ffmpeg.exe
os.environ["PATH"] += os.pathsep + os.getcwd()

import whisper
from utils import load_settings, get_data_paths, setup_logging, timestamp

logger = setup_logging("transcribe", log_file="logs/transcribe.log")


def transcribe_file(model, audio_path, language="ur"):
    """Transcribe a single audio file. Returns dict with text and metadata."""
    try:
        result = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            verbose=False,
        )
        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", language),
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"].strip(),
                }
                for s in result.get("segments", [])
            ],
        }
    except Exception as e:
        logger.error("  Failed to transcribe %s: %s", audio_path.name, e)
        return None


def run_transcription(model_name=None, device=None):
    """Transcribe all processed audio segments."""
    settings = load_settings()
    paths = get_data_paths(settings)
    t_cfg = settings.get("transcription", {})

    model_name = model_name or t_cfg.get("model_name", "large-v3")
    device = device or t_cfg.get("device", "cuda")
    language = t_cfg.get("language", "ur")

    proc_dir = paths["processed"]
    ann_dir = paths["annotations"]

    wav_files = sorted(proc_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No WAV files in %s. Run preprocess_audio.py first.", proc_dir)
        return

    logger.info("Loading Whisper model '%s' on %s...", model_name, device)
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        logger.error("Failed to load Whisper model: %s", e)
        logger.info("Trying CPU fallback...")
        model = whisper.load_model(model_name, device="cpu")

    logger.info("Transcribing %d segment(s)", len(wav_files))
    ok, skip, fail = 0, 0, 0

    for f in wav_files:
        seg_id = f.stem
        out_path = ann_dir / f"{seg_id}_transcript.json"

        if out_path.exists():
            logger.info("  ⏭ %s already transcribed", seg_id)
            skip += 1
            continue

        logger.info("  🎤 Transcribing: %s", seg_id)
        result = transcribe_file(model, f, language)

        if result:
            result["segment_id"] = seg_id
            result["source_file"] = f.name
            result["model"] = model_name
            result["transcribed_at"] = timestamp()

            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(result, fp, indent=2, ensure_ascii=False)
            ok += 1
            preview = result["text"][:80] + "..." if len(result["text"]) > 80 else result["text"]
            logger.info("  ✅ %s: '%s'", seg_id, preview)
        else:
            fail += 1

    logger.info("Done: %d transcribed, %d skipped, %d failed", ok, skip, fail)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Transcribe Urdu audio with Whisper.")
    p.add_argument("--model", default=None, help="Whisper model size.")
    p.add_argument("--device", default=None, help="Device: cuda or cpu.")
    args = p.parse_args()
    run_transcription(model_name=args.model, device=args.device)
