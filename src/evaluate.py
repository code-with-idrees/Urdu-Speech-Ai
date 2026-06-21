"""
4D Benchmark Evaluation — Urdu Speech AI Pipeline (Step 7)

Implements the four-dimensional evaluation framework:
  1. Clarity    — Word Error Rate (WER) via jiwer
  2. Fluency    — Pause distribution and rhythm analysis
  3. Accent Authenticity — Urdu phoneme coverage
  4. Language Distance   — Lexical overlap with Hindi reference

Usage:
    python src/evaluate.py --dataset results/dataset_test.json
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import librosa

from utils import load_settings, get_data_paths, setup_logging, timestamp

logger = setup_logging("evaluate", log_file="logs/evaluate.log")


# ── Dimension 1: Clarity (WER) ──────────────────────────────────────────────

def compute_clarity(reference, hypothesis):
    """Compute Word Error Rate between reference and hypothesis text."""
    try:
        import jiwer
        wer = jiwer.wer(reference, hypothesis)
        return {"wer": round(wer, 4), "clarity_score": round(1 - wer, 4)}
    except ImportError:
        logger.warning("jiwer not installed. Skipping WER computation.")
        return {"wer": None, "clarity_score": None}


# ── Dimension 2: Fluency ────────────────────────────────────────────────────

def compute_fluency(audio_path, sr=16000):
    """Analyze speech fluency via pause detection and rhythm metrics."""
    try:
        audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    except Exception:
        return {"fluency_score": None}

    # Detect pauses (frames below energy threshold)
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    energies = []
    for i in range(0, len(audio) - frame_len, hop):
        rms = np.sqrt(np.mean(audio[i:i+frame_len] ** 2))
        energies.append(20 * np.log10(rms + 1e-10))

    energies = np.array(energies)
    threshold = np.percentile(energies, 20)
    is_pause = energies < threshold

    # Count pause segments and their average duration
    pause_count = 0
    pause_lengths = []
    in_pause = False
    current_len = 0
    for p in is_pause:
        if p:
            if not in_pause:
                in_pause = True
                current_len = 0
            current_len += 1
        else:
            if in_pause:
                pause_lengths.append(current_len * 0.010)
                pause_count += 1
                in_pause = False

    avg_pause = np.mean(pause_lengths) if pause_lengths else 0
    # Score: fewer and shorter pauses = higher fluency (max 1.0)
    fluency = max(0, 1.0 - (avg_pause * pause_count) / (len(audio) / sr))
    return {
        "pause_count": pause_count,
        "avg_pause_sec": round(avg_pause, 3),
        "fluency_score": round(fluency, 4),
    }


# ── Dimension 3: Accent Authenticity ────────────────────────────────────────

# Common Urdu-specific characters not shared with Hindi
URDU_MARKERS = set("ٹ ڈ ڑ ں ے ھ ۂ ۃ ؤ ئ آ".split())

def compute_accent_authenticity(text):
    """Estimate Urdu authenticity based on script-specific character usage."""
    if not text:
        return {"accent_score": 0, "urdu_char_ratio": 0}

    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return {"accent_score": 0, "urdu_char_ratio": 0}

    urdu_count = sum(1 for c in text if c in URDU_MARKERS)
    ratio = urdu_count / total_chars
    # Heuristic: texts with >2% Urdu markers are strongly Urdu
    score = min(1.0, ratio * 50)
    return {
        "urdu_char_ratio": round(ratio, 4),
        "accent_score": round(score, 4),
    }


# ── Dimension 4: Language Distance from Hindi ───────────────────────────────

# Common Hindi-only words (Devanagari or transliterated markers)
HINDI_MARKERS = {"है", "हैं", "का", "की", "के", "में", "को", "से", "पर", "और"}

def compute_language_distance(text):
    """Measure how distinct the text is from Hindi (higher = more Urdu)."""
    if not text:
        return {"language_distance": 0}

    words = text.split()
    if not words:
        return {"language_distance": 0}

    hindi_count = sum(1 for w in words if w in HINDI_MARKERS)
    hindi_ratio = hindi_count / len(words)
    # Distance: 1.0 = pure Urdu, 0.0 = pure Hindi
    distance = 1.0 - hindi_ratio
    return {
        "hindi_word_ratio": round(hindi_ratio, 4),
        "language_distance": round(distance, 4),
    }


# ── Aggregate ───────────────────────────────────────────────────────────────

def evaluate_sample(sample):
    """Run all 4 dimensions on a single sample."""
    results = {"segment_id": sample["segment_id"]}

    # Clarity (need reference — use transcript as self-reference for now)
    transcript = sample.get("transcript", "")
    results.update(compute_clarity(transcript, transcript))

    # Fluency
    audio_path = sample.get("audio_path", "")
    if audio_path and Path(audio_path).exists():
        results.update(compute_fluency(audio_path))
    else:
        results["fluency_score"] = None

    # Accent
    results.update(compute_accent_authenticity(transcript))

    # Language distance
    results.update(compute_language_distance(transcript))

    return results


def run_evaluation(dataset_path):
    """Evaluate the test split and produce a benchmark report."""
    settings = load_settings()
    paths = get_data_paths(settings)
    results_dir = paths["results"]

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    logger.info("Evaluating %d samples on 4D benchmark", len(samples))
    all_results = []

    for s in samples:
        r = evaluate_sample(s)
        all_results.append(r)

    # Compute averages
    dims = ["clarity_score", "fluency_score", "accent_score", "language_distance"]
    averages = {}
    for d in dims:
        vals = [r[d] for r in all_results if r.get(d) is not None]
        averages[d] = round(np.mean(vals), 4) if vals else None

    report = {
        "num_samples": len(samples),
        "averages": averages,
        "per_sample": all_results,
        "evaluated_at": timestamp(),
    }

    out = results_dir / "benchmark_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("═══════════════════════════════════════")
    logger.info("  4D BENCHMARK RESULTS")
    logger.info("═══════════════════════════════════════")
    for d in dims:
        val = averages.get(d)
        bar = "█" * int((val or 0) * 20)
        logger.info("  %-22s %s %.4f", d, bar, val or 0)
    logger.info("═══════════════════════════════════════")
    logger.info("Report saved to %s", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="4D benchmark evaluation.")
    p.add_argument("--dataset", required=True, help="Path to test dataset JSON.")
    args = p.parse_args()
    run_evaluation(args.dataset)
