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

# Audience reaction lexical and confidence thresholds
AUDIENCE_KEYWORDS = [
    # Urdu
    "واہ", "کیا کہنے", "سبحان اللہ", "بہت خوب", "مکرر", "ملاحظہ", "زندہ باد", "جیو",
    "کیا بات ہے", "ماشاء اللہ", "ماشاءاللہ", "شاباش",
    # Roman Urdu / English
    "wah", "kya kehne", "subhanallah", "clapping", "applause", "laughter", "cheering",
    "kya baat hai", "kya bat hai", "bohat khoob", "bahut khoob", "bohat khub", "bahut khub",
    "masha allah", "mashallah", "mashaallah", "shabash", "shabaash"
]


def check_audience_reaction(text, avg_logprob, no_speech_prob):
    """Determine if a transcribed segment is likely an audience reaction.
    
    Returns (is_reaction, reason).
    """
    # 1. Check no-speech probability (Whisper detected silence or non-speech)
    if no_speech_prob > 0.5:
        return True, f"high no_speech_prob ({no_speech_prob:.2f})"
        
    # 2. Check average log probability (low confidence = background chatter/noise)
    if avg_logprob < -1.1:
        return True, f"low avg_logprob ({avg_logprob:.2f})"
        
    # 3. Check lexical markers
    cleaned = "".join(c for c in text.lower() if c.isalnum() or c.isspace()).strip()
    words = cleaned.split()
    if not words:
        return True, "empty transcript"
        
    # Count how many words contain or match audience keywords
    matches = sum(1 for w in words if any(kw in w for kw in AUDIENCE_KEYWORDS))
    match_ratio = matches / len(words)
    
    # If more than 40% of the words are audience cheers
    if match_ratio > 0.40:
        return True, f"lexical match ratio ({match_ratio:.2f}) on text: '{text}'"
        
    return False, ""


def transcribe_file(model, audio_path, language="ur"):
    """Transcribe a single audio file. Returns dict with text and metadata."""
    try:
        result = model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            verbose=False,
        )
        
        segments = result.get("segments", [])
        avg_logprob = sum(s.get("avg_logprob", 0.0) for s in segments) / len(segments) if segments else -99.0
        no_speech_prob = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments) if segments else 0.0

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", language),
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "segments": [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "text": s["text"].strip(),
                    "avg_logprob": s.get("avg_logprob", 0.0),
                    "no_speech_prob": s.get("no_speech_prob", 0.0),
                }
                for s in segments
            ],
        }
    except Exception as e:
        logger.error("  Failed to transcribe %s: %s", audio_path.name, e)
        return None


def run_transcription(model_name=None, device=None, limit=None):
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

    if limit is not None:
        logger.info("Limiting transcription to first %d segment(s).", limit)
        wav_files = wav_files[:limit]

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
            # Check if this segment contains audience reactions (e.g. wah wah)
            is_reaction, reason = check_audience_reaction(
                result["text"], result["avg_logprob"], result["no_speech_prob"]
            )
            if is_reaction:
                logger.warning("  ❌ Rejected segment %s as audience reaction: %s", seg_id, reason)
                # Move the rejected audio file to data/discarded_segments to clean the dataset
                discard_dir = proc_dir.parent / "discarded_segments"
                discard_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Move file (needs to handle existing file)
                    dest_file = discard_dir / f.name
                    if dest_file.exists():
                        dest_file.unlink()
                    f.rename(dest_file)
                except Exception as e:
                    logger.error("    Failed to move discarded segment: %s", e)
                fail += 1
                continue

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
    p.add_argument("--limit", type=int, default=None, help="Limit number of files to transcribe.")
    args = p.parse_args()
    run_transcription(model_name=args.model, device=args.device, limit=args.limit)
