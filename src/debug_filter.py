"""Quick diagnostic: re-test 10 segments with FIXED thresholds."""
import sys, json
from pathlib import Path
import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from filter_audience import (
    compute_energy_features, compute_spectral_features,
    compute_vad_features, detect_sudden_bursts, classify_segment, THRESHOLDS,
)

segments_dir = Path(r"d:\Urdu-Speech-AI\data\segments")
files = sorted(segments_dir.glob("*.wav"))[:20]
t = THRESHOLDS["normal"]

print(f"Sampling {len(files)} segments from {segments_dir}")
print(f"Thresholds (normal): {json.dumps(t, indent=2)}")
print("=" * 90)

clean_count = 0
rejected_count = 0

for f in files:
    audio, sr = librosa.load(str(f), sr=16000, mono=True)
    dur = len(audio) / sr
    feats = {"filename": f.name, "duration_sec": dur}
    feats.update(compute_energy_features(audio, sr))
    feats.update(compute_spectral_features(audio, sr))
    feats.update(compute_vad_features(audio, sr))
    feats.update(detect_sudden_bursts(audio, sr))

    label, reasons = classify_segment(feats, t)
    
    if label == "clean":
        clean_count += 1
    else:
        rejected_count += 1

    print(f"\n--- {f.name} ({dur:.1f}s) => {label.upper()} ---")
    print(f"  energy_std_db:       {feats['energy_std_db']:8.2f}  (max: {t['energy_std_max_db']})")
    print(f"  spectral_flatness:   {feats['spectral_flatness']:8.4f}  (min: {t['spectral_flatness_min']}, max: {t['spectral_flatness_max']})")
    print(f"  zero_crossing_rate:  {feats['zero_crossing_rate']:8.4f}  (max: {t['zcr_max']})")
    print(f"  speech_ratio:        {feats['speech_ratio']:8.4f}  (min: {t['min_speech_ratio']}, max: {t['max_speech_ratio']})")
    print(f"  energy_mean_db:      {feats['energy_mean_db']:8.2f}  (silence: {t['rms_silence_threshold_db']})")
    print(f"  burst_rate/sec:      {feats['burst_rate_per_sec']:8.4f}  (max: {t['burst_rate_max_per_sec']})")
    if reasons:
        print(f"  REJECTION REASONS: {reasons}")

print(f"\n{'=' * 90}")
print(f"RESULTS: {clean_count} clean, {rejected_count} rejected out of {len(files)}")
print(f"Clean rate: {clean_count / len(files) * 100:.0f}%")
