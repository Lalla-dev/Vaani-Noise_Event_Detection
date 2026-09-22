"""
cache_gold_subset.py
Standalone script to pre-cache the first 1,000 Gold training clips
from the ARTPARK-IISc/Vaani-Noise-Event-Dataset using a single sequential
Hugging Face streaming pass.

Features:
- Selects the first 1,000 Gold clips sorted by dataset index.
- Automatically skips clips that are already cached in data/cache/audio_clips/.
- Streams sequentially from index 0 up to max index 2859, then stops.
- Avoids repetitive .skip() overhead and torchcodec dependencies.
- Verifies all 1,000 target files exist and reports cache statistics.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Ensure Windows-safe console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset
from dotenv import load_dotenv

import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Pre-cache Gold audio clips from HuggingFace dataset")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of Gold samples to pre-cache (default: 1000)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"VAANI NOISE EVENT DETECTION - GOLD SUBSET PRE-CACHING ({args.num_samples} CLIPS)")
    print("=" * 60)

    train_jsonl = PROJECT_ROOT / "data" / "processed" / "train_split.jsonl"
    cache_dir = PROJECT_ROOT / "data" / "cache" / "audio_clips"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and identify Gold samples sorted by index
    print(f"Reading {train_jsonl} to identify first {args.num_samples} Gold records...")
    gold_samples = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("tier") == "gold":
                try:
                    idx = int(rec["clip_id"].replace("train_", ""))
                    gold_samples.append((idx, rec["clip_id"]))
                except ValueError:
                    continue

    gold_samples.sort(key=lambda x: x[0])
    target_subset = gold_samples[:args.num_samples]
    target_indices = {idx: cid for idx, cid in target_subset}

    min_idx = target_subset[0][0]
    max_idx = target_subset[-1][0]
    print(f"Total Gold records in train_split: {len(gold_samples)}")
    print(f"Selected target subset: {len(target_subset)} clips")
    print(f"Index range: [{min_idx} ('{target_subset[0][1]}') to {max_idx} ('{target_subset[-1][1]}')]")

    # 2. Check which target clips are already cached
    existing_files = set(f.name for f in cache_dir.glob("*.wav") if f.stat().st_size > 0)
    needed_indices = {}
    already_cached_count = 0

    for idx, cid in target_indices.items():
        fname = f"{cid}.wav"
        if fname in existing_files:
            already_cached_count += 1
        else:
            needed_indices[idx] = cid

    print(f"Already cached: {already_cached_count}/{len(target_indices)}")
    print(f"Remaining to download and cache: {len(needed_indices)}/{len(target_indices)}")

    if not needed_indices:
        print(f"\nAll {len(target_indices)} target Gold clips are already cached! No streaming needed.")
    else:
        # 3. Initialize Hugging Face streaming dataset
        load_dotenv(PROJECT_ROOT / ".env")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not found in .env or environment variables.")

        print("\nInitializing Hugging Face streaming dataset (decode=False)...")
        t_init = time.time()
        ds = load_dataset(
            "ARTPARK-IISc/Vaani-Noise-Event-Dataset",
            token=token,
            streaming=True,
        )
        train_stream = ds["train"].cast_column("audio", Audio(decode=False))
        print(f"Streaming dataset initialized in {time.time() - t_init:.2f}s")

        # 4. Single sequential streaming pass
        print(f"\nStarting sequential streaming pass (indices 0 to {max_idx})...")
        t_start = time.time()
        cached_in_run = 0
        target_sr = 16000

        for idx, item in enumerate(train_stream):
            if idx > max_idx:
                print(f"\nReached max target index {max_idx} (current stream index {idx}). Stopping stream.")
                break

            if idx in needed_indices:
                cid = needed_indices[idx]
                cache_path = cache_dir / f"{cid}.wav"

                raw_bytes = item["audio"]["bytes"]
                audio_array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")

                if sr != target_sr:
                    audio_tensor = torch.from_numpy(audio_array).float()
                    if audio_tensor.ndim == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    audio_array = resampler(audio_tensor).squeeze().numpy()
                    sr = target_sr

                sf.write(cache_path, audio_array, target_sr, subtype="PCM_16")
                cached_in_run += 1

                if cached_in_run % 50 == 0 or (already_cached_count + cached_in_run) == len(target_indices):
                    elapsed = time.time() - t_start
                    total_done = already_cached_count + cached_in_run
                    rate = cached_in_run / max(elapsed, 1e-3)
                    print(
                        f"[{total_done:5d}/{len(target_indices)}] Cached {cid} (idx {idx:5d}) | "
                        f"Run: {cached_in_run} clips in {elapsed:.1f}s ({rate:.1f} clips/s)",
                        flush=True,
                    )

            if (already_cached_count + cached_in_run) >= len(target_indices):
                print(f"\nAll {len(target_indices)} target Gold clips have been cached! Stopping stream early.")
                break

        total_elapsed = time.time() - t_start
        print(f"\nStreaming pass finished in {total_elapsed:.1f}s. Newly cached: {cached_in_run} clips.")

    # 5. Verification
    print("\n" + "=" * 60)
    print(f"VERIFICATION OF CACHED {args.num_samples} GOLD SAMPLES")
    print("=" * 60)

    missing = []
    target_sizes = []
    for idx, cid in target_subset:
        p = cache_dir / f"{cid}.wav"
        if not p.exists() or p.stat().st_size == 0:
            missing.append(cid)
        else:
            target_sizes.append(p.stat().st_size)

    if missing:
        print(f"WARNING: {len(missing)} target clips are missing from cache!")
        print(f"Missing sample IDs: {missing[:10]}")
    else:
        print(f"SUCCESS: All {len(target_subset)} target Gold WAV files exist in {cache_dir}!")

    total_target_bytes = sum(target_sizes)
    all_cached_files = list(cache_dir.glob("*.wav"))
    total_all_bytes = sum(f.stat().st_size for f in all_cached_files)

    print(f"Total target Gold clips verified: {len(target_sizes)}/{args.num_samples}")
    print(f"Total target audio size: {total_target_bytes / (1024 * 1024):.2f} MB")
    print(f"Total cache directory files: {len(all_cached_files)} WAVs")
    print(f"Total cache directory size: {total_all_bytes / (1024 * 1024):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
