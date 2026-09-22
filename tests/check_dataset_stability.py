"""
check_dataset_stability.py

Stability Check: Verify karo ki HuggingFace dataset streaming mein:
1. Same index par same sample milta hai ya nahi (sequential stability)
2. Koi unique stable field available hai jo hamesha constant rahe (like imageFileName)
3. Dataset ordering consistent hai ya nahi multiple streams mein

Usage:
    python check_dataset_stability.py
"""

import os
import json
from datasets import load_dataset, Audio
from dotenv import load_dotenv
from collections import Counter

print("=" * 70)
print("HUGGING FACE DATASET - STABILITY CHECK")
print("=" * 70)

# Load HF token
load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    print("❌ HF_TOKEN not found in .env file")
    exit(1)

print(f"\n✓ HF_TOKEN found (length: {len(token)})")

# Dataset name
DATASET_NAME = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"

print(f"\n📦 Loading dataset: {DATASET_NAME}")
print("   Mode: Streaming (for stability test)")

# ============================================================================
# TEST 1: Check available fields in the dataset
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: Checking Available Fields")
print("=" * 70)

try:
    dataset = load_dataset(DATASET_NAME, token=token, streaming=True)
    train_split = dataset["train"].cast_column("audio", Audio(decode=False))
    
    # Get first sample
    first_sample = next(iter(train_split))
    
    print("\n✓ Available fields in dataset:")
    for i, key in enumerate(first_sample.keys(), 1):
        value = first_sample[key]
        value_preview = str(value)[:100] if not isinstance(value, dict) else "[dict]"
        print(f"   {i:2d}. {key:<30} = {value_preview}")
    
    print(f"\n✓ Total fields: {len(first_sample.keys())}")
    
    # Check for potential unique identifiers
    print("\n🔍 Checking for unique identifier fields:")
    unique_candidates = []
    
    for key in first_sample.keys():
        # Check if field name suggests it's an ID
        if any(id_term in key.lower() for id_term in ['id', 'filename', 'name', 'path', 'url']):
            unique_candidates.append(key)
            print(f"   ✓ Potential ID field: {key} = {first_sample[key]}")
    
    if not unique_candidates:
        print("   ⚠ No obvious ID fields found (will rely on sequential index)")
    
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    exit(1)

# ============================================================================
# TEST 2: Sequential Stability Check
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Sequential Stability Check")
print("=" * 70)
print("\nTesting if same index returns same sample across multiple streams...")

# Test indices to check
test_indices = [0, 1, 10, 100, 1000]

print(f"\nTest strategy: Stream dataset 2 times and compare samples at indices: {test_indices}")

def get_samples_at_indices(indices):
    """Stream dataset and collect samples at specified indices"""
    dataset = load_dataset(DATASET_NAME, token=token, streaming=True)
    train_split = dataset["train"].cast_column("audio", Audio(decode=False))
    
    samples = {}
    max_index = max(indices)
    
    for i, sample in enumerate(train_split):
        if i in indices:
            # Store relevant fields (not audio data, just metadata)
            samples[i] = {
                'transcript': sample.get('transcript', ''),
                'duration': sample.get('duration', None),
                'language': sample.get('language', ''),
                'state': sample.get('state', ''),
                'annotationQuality': sample.get('annotationQuality', ''),
            }
            
            # Also store any ID-like fields
            for key in sample.keys():
                if any(id_term in key.lower() for id_term in ['id', 'filename', 'name', 'path']):
                    samples[i][f'id_field_{key}'] = sample.get(key, '')
        
        # Stop after we've collected all needed samples
        if i > max_index:
            break
    
    return samples

print("\n🔄 Stream 1: Collecting samples...")
stream1_samples = get_samples_at_indices(test_indices)

print("🔄 Stream 2: Collecting samples again...")
stream2_samples = get_samples_at_indices(test_indices)

print("\n📊 Comparing samples from both streams:\n")

all_stable = True
for idx in test_indices:
    if idx not in stream1_samples or idx not in stream2_samples:
        print(f"Index {idx}: ❌ NOT FOUND in one or both streams")
        all_stable = False
        continue
    
    s1 = stream1_samples[idx]
    s2 = stream2_samples[idx]
    
    # Compare all fields
    is_same = True
    mismatches = []
    
    for key in s1.keys():
        if s1[key] != s2.get(key):
            is_same = False
            mismatches.append(key)
    
    if is_same:
        print(f"Index {idx:4d}: ✓ STABLE")
        print(f"           Language: {s1['language']}, Duration: {s1['duration']}s")
        print(f"           Transcript: {s1['transcript'][:60]}...")
    else:
        print(f"Index {idx:4d}: ❌ UNSTABLE - Fields differ: {mismatches}")
        all_stable = False
    
    print()

# ============================================================================
# TEST 3: Check if there's a stable unique identifier field
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: Unique Identifier Field Check")
print("=" * 70)

print("\nChecking first 100 samples for unique identifier patterns...")

dataset = load_dataset(DATASET_NAME, token=token, streaming=True)
train_split = dataset["train"].cast_column("audio", Audio(decode=False))

# Collect first 100 samples
samples_for_id_check = []
for i, sample in enumerate(train_split):
    if i >= 100:
        break
    samples_for_id_check.append(sample)

print(f"\n✓ Collected {len(samples_for_id_check)} samples")

# Check each field for uniqueness
print("\n🔍 Analyzing fields for uniqueness:\n")

for key in samples_for_id_check[0].keys():
    if key == 'audio':  # Skip audio field
        continue
    
    # Collect all values for this field
    values = []
    for sample in samples_for_id_check:
        val = sample.get(key)
        if val is not None and not isinstance(val, (dict, list)):
            values.append(str(val))
    
    if not values:
        continue
    
    # Check uniqueness
    unique_values = set(values)
    uniqueness_ratio = len(unique_values) / len(values)
    
    # A field is a good candidate if it has high uniqueness
    if uniqueness_ratio > 0.8:
        print(f"Field: {key:<30}")
        print(f"  Unique values: {len(unique_values)}/{len(values)} ({uniqueness_ratio*100:.1f}%)")
        print(f"  Sample values: {values[:3]}")
        
        if uniqueness_ratio == 1.0:
            print(f"  ✅ 100% UNIQUE - Perfect candidate for stable mapping!")
        else:
            print(f"  ⚠ {uniqueness_ratio*100:.1f}% unique - May have duplicates")
        print()

# ============================================================================
# TEST 4: Match with our unified.jsonl
# ============================================================================
print("\n" + "=" * 70)
print("TEST 4: Cross-Verification with unified.jsonl")
print("=" * 70)

print("\nChecking if our unified.jsonl samples match HF dataset at same indices...")

# Load first 10 samples from unified.jsonl
unified_samples = []
try:
    with open("data/processed/unified.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            unified_samples.append(json.loads(line))
    
    print(f"\n✓ Loaded {len(unified_samples)} samples from unified.jsonl")
except Exception as e:
    print(f"\n❌ Error loading unified.jsonl: {e}")
    unified_samples = []

if unified_samples:
    # Stream HF dataset and compare
    dataset = load_dataset(DATASET_NAME, token=token, streaming=True)
    train_split = dataset["train"].cast_column("audio", Audio(decode=False))
    
    print("\n📊 Comparing unified.jsonl with live HF stream:\n")
    
    matches = 0
    for i, hf_sample in enumerate(train_split):
        if i >= len(unified_samples):
            break
        
        unified = unified_samples[i]
        
        # Compare key fields
        transcript_match = unified['transcript'] == hf_sample.get('transcript', '')
        duration_match = abs(unified['duration'] - hf_sample.get('duration', 0)) < 0.01
        language_match = unified['language'] == hf_sample.get('language', '')
        
        if transcript_match and duration_match and language_match:
            matches += 1
            print(f"Index {i}: ✓ MATCH")
        else:
            print(f"Index {i}: ❌ MISMATCH")
            print(f"  Transcript match: {transcript_match}")
            print(f"  Duration match: {duration_match}")
            print(f"  Language match: {language_match}")
    
    print(f"\n✓ Matches: {matches}/{len(unified_samples)} ({matches/len(unified_samples)*100:.1f}%)")
    
    if matches == len(unified_samples):
        print("\n🎉 Perfect match! Sequential indexing is STABLE!")
    else:
        print("\n⚠ Some mismatches found. May need alternative mapping strategy.")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RECOMMENDATION")
print("=" * 70)

if all_stable:
    print("\n✅ RESULT: Sequential indexing is STABLE")
    print("\nRecommendation:")
    print("  • Use sequential index mapping: clip_id → HF index")
    print("  • train_000042 → index 42 is RELIABLE")
    print("  • Current audio_loader.py approach is CORRECT")
    print("\nYour audio loading system is READY TO USE! 🎉")
else:
    print("\n⚠ RESULT: Sequential indexing may be UNSTABLE")
    print("\nRecommendation:")
    print("  • Check if a unique ID field is available")
    print("  • Consider using a stable mapping file")
    print("  • May need to modify audio_loader.py strategy")

print("\n" + "=" * 70)
