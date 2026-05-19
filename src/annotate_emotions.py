"""
AI-Assisted Emotion Annotation — Urdu Speech AI Pipeline (Step 5)

Uses Google Gemini to classify the primary emotion of each transcribed
audio segment. Reads transcripts from data/annotations/, sends them to
Gemini with a structured prompt, and saves emotion labels as JSON.

Usage:
    python src/annotate_emotions.py
    python src/annotate_emotions.py --dry-run    # preview prompts only
"""

import argparse
import json
import time
from pathlib import Path

import google.generativeai as genai

from utils import (
    load_settings, get_data_paths, get_emotion_ids, load_emotions,
    get_gemini_api_key, setup_logging, timestamp,
)

logger = setup_logging("annotate", log_file="logs/annotate_emotions.log")

PROMPT_TEMPLATE = """You are an expert in Urdu poetry (Shayari) and emotional analysis.
You understand the deep cultural and linguistic nuances of Urdu ghazals, nazms, and mushaira performances.

Given this transcript of an Urdu poetry performance:

---
{transcript}
---

Classify the PRIMARY emotion expressed in this performance from EXACTLY one of these categories:
{emotion_list}

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "primary_emotion": "<emotion_id>",
  "confidence": <0.0 to 1.0>,
  "secondary_emotion": "<emotion_id or null>",
  "reasoning": "<1-2 sentence explanation in English>"
}}
"""


def build_prompt(transcript, emotions):
    """Build the classification prompt with the transcript and emotion list."""
    emo_lines = "\n".join(
        f"  - {e['id']}: {e['description']}" for e in emotions
    )
    return PROMPT_TEMPLATE.format(transcript=transcript, emotion_list=emo_lines)


def classify_emotion(model, transcript, emotions, temperature=0.3):
    """Send transcript to Gemini and parse the emotion classification."""
    prompt = build_prompt(transcript, emotions)
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=300,
            ),
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        result = json.loads(text)
        valid_ids = [e["id"] for e in emotions]
        if result.get("primary_emotion") not in valid_ids:
            logger.warning("  Invalid emotion '%s', skipping", result.get("primary_emotion"))
            return None
        return result
    except json.JSONDecodeError as e:
        logger.error("  JSON parse error: %s | Raw: %s", e, text[:200])
        return None
    except Exception as e:
        logger.error("  Gemini API error: %s", e)
        return None


def run_annotation(dry_run=False):
    """Annotate all transcribed segments with emotions via Gemini."""
    settings = load_settings()
    paths = get_data_paths(settings)
    ann_cfg = settings.get("annotation", {})
    emotions = load_emotions()

    ann_dir = paths["annotations"]
    min_conf = ann_cfg.get("min_confidence", 0.6)
    temperature = ann_cfg.get("temperature", 0.3)
    max_retries = ann_cfg.get("max_retries", 3)

    # Find all transcripts
    transcripts = sorted(ann_dir.glob("*_transcript.json"))
    if not transcripts:
        logger.warning("No transcripts found in %s. Run transcribe.py first.", ann_dir)
        return

    if not dry_run:
        api_key = get_gemini_api_key()
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(ann_cfg.get("gemini_model", "gemini-2.0-flash"))
    else:
        gemini_model = None

    logger.info("Annotating %d transcript(s) (dry_run=%s)", len(transcripts), dry_run)
    ok, skip, fail, low_conf = 0, 0, 0, 0

    for t_path in transcripts:
        seg_id = t_path.stem.replace("_transcript", "")
        label_path = ann_dir / f"{seg_id}_emotion.json"

        if label_path.exists():
            logger.info("  ⏭ %s already annotated", seg_id)
            skip += 1
            continue

        with open(t_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        transcript = t_data.get("text", "")
        if not transcript.strip():
            logger.warning("  ⚠ Empty transcript for %s", seg_id)
            fail += 1
            continue

        if dry_run:
            prompt = build_prompt(transcript, emotions)
            logger.info("  [DRY RUN] %s:\n%s\n", seg_id, prompt[:300])
            continue

        # Retry loop
        result = None
        for attempt in range(1, max_retries + 1):
            result = classify_emotion(gemini_model, transcript, emotions, temperature)
            if result:
                break
            logger.info("  Retry %d/%d for %s", attempt, max_retries, seg_id)
            time.sleep(2 * attempt)

        if not result:
            fail += 1
            continue

        conf = result.get("confidence", 0)
        if conf < min_conf:
            logger.info("  ⚠ Low confidence (%.2f) for %s", conf, seg_id)
            low_conf += 1

        label_data = {
            "segment_id": seg_id,
            "primary_emotion": result["primary_emotion"],
            "secondary_emotion": result.get("secondary_emotion"),
            "confidence": conf,
            "reasoning": result.get("reasoning", ""),
            "model": ann_cfg.get("gemini_model", "gemini-2.0-flash"),
            "human_verified": False,
            "annotated_at": timestamp(),
        }
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label_data, f, indent=2, ensure_ascii=False)
        ok += 1
        logger.info("  ✅ %s → %s (%.0f%%)", seg_id, result["primary_emotion"], conf * 100)

        time.sleep(0.5)  # Rate limiting

    logger.info("Done: %d annotated, %d skipped, %d failed, %d low-confidence", ok, skip, fail, low_conf)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AI-assisted emotion annotation via Gemini.")
    p.add_argument("--dry-run", action="store_true", help="Preview prompts without calling API.")
    args = p.parse_args()
    run_annotation(dry_run=args.dry_run)
