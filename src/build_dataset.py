"""
Dataset Builder — Urdu Speech AI Pipeline (Step 6)

Assembles processed audio + transcripts + emotion labels into a unified
HuggingFace-compatible dataset with train/val/test splits.

Usage:
    python src/build_dataset.py
    python src/build_dataset.py --format csv   # export as CSV instead
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd

from utils import load_settings, get_data_paths, setup_logging, timestamp

logger = setup_logging("build_dataset", log_file="logs/build_dataset.log")


def collect_samples(proc_dir, ann_dir):
    """Gather all samples that have audio + transcript + emotion label."""
    samples = []
    emotion_files = sorted(ann_dir.glob("*_emotion.json"))

    for emo_path in emotion_files:
        seg_id = emo_path.stem.replace("_emotion", "")
        audio_path = proc_dir / f"{seg_id}.wav"
        transcript_path = ann_dir / f"{seg_id}_transcript.json"

        if not audio_path.exists():
            logger.warning("  Missing audio for %s", seg_id)
            continue
        if not transcript_path.exists():
            logger.warning("  Missing transcript for %s", seg_id)
            continue

        with open(emo_path, "r", encoding="utf-8") as f:
            emo = json.load(f)
        with open(transcript_path, "r", encoding="utf-8") as f:
            trans = json.load(f)

        samples.append({
            "segment_id": seg_id,
            "audio_path": str(audio_path),
            "transcript": trans.get("text", ""),
            "poet": emo.get("poet", seg_id.rsplit("_", 2)[0]),
            "primary_emotion": emo.get("primary_emotion", ""),
            "secondary_emotion": emo.get("secondary_emotion"),
            "confidence": emo.get("confidence", 0),
            "human_verified": emo.get("human_verified", False),
        })

    return samples


def split_dataset(samples, train_r=0.8, val_r=0.1, test_r=0.1, seed=42):
    """Split samples into train/val/test ensuring emotion stratification."""
    import random
    random.seed(seed)

    # Group by emotion for stratified split
    by_emotion = {}
    for s in samples:
        emo = s["primary_emotion"]
        by_emotion.setdefault(emo, []).append(s)

    train, val, test = [], [], []
    for emo, emo_samples in by_emotion.items():
        random.shuffle(emo_samples)
        n = len(emo_samples)
        n_train = max(1, int(n * train_r))
        n_val = max(1, int(n * val_r))

        train.extend(emo_samples[:n_train])
        val.extend(emo_samples[n_train:n_train + n_val])
        test.extend(emo_samples[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def run_build(output_format="huggingface"):
    """Build the final dataset."""
    settings = load_settings()
    paths = get_data_paths(settings)
    ds_cfg = settings.get("dataset", {})

    proc_dir = paths["processed"]
    ann_dir = paths["annotations"]
    results_dir = paths["results"]

    samples = collect_samples(proc_dir, ann_dir)
    if not samples:
        logger.warning("No complete samples found. Run the full pipeline first.")
        return

    logger.info("Found %d complete samples", len(samples))

    # Print emotion distribution
    dist = Counter(s["primary_emotion"] for s in samples)
    logger.info("Emotion distribution:")
    for emo, count in dist.most_common():
        logger.info("  %-15s %d", emo, count)

    # Split
    train, val, test = split_dataset(
        samples,
        ds_cfg.get("train_split", 0.8),
        ds_cfg.get("val_split", 0.1),
        ds_cfg.get("test_split", 0.1),
    )
    logger.info("Splits: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    # Save
    if output_format == "csv":
        for name, data in [("train", train), ("val", val), ("test", test)]:
            df = pd.DataFrame(data)
            out = results_dir / f"dataset_{name}.csv"
            df.to_csv(out, index=False, encoding="utf-8")
            logger.info("Saved %s (%d rows)", out.name, len(df))
    else:
        # Save as JSON (HuggingFace datasets can load this)
        for name, data in [("train", train), ("val", val), ("test", test)]:
            out = results_dir / f"dataset_{name}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("Saved %s (%d samples)", out.name, len(data))

    # Save summary stats
    summary = {
        "total_samples": len(samples),
        "train": len(train), "val": len(val), "test": len(test),
        "emotion_distribution": dict(dist),
        "built_at": timestamp(),
    }
    with open(results_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("✅ Dataset built successfully in %s", results_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the final annotated dataset.")
    p.add_argument("--format", default="huggingface", choices=["huggingface", "csv"])
    args = p.parse_args()
    run_build(output_format=args.format)
