"""
cache_silver_subset.py
Dedicated script to select, filter, and pre-cache a deterministic Silver-tier
training subset from ARTPARK-IISc/Vaani-Noise-Event-Dataset using Hugging Face streaming.

Features:
- Selects exclusively Silver records (tier == "silver") from train_split.jsonl.
- Filters out blanket noise annotations spanning >90% of clip duration.
- Filters out invalid timestamps (negative, inverted, or empty events).
- Deterministically selects the first N clean Silver clips sorted by dataset index.
- Generates a unified manifest: data/processed/train_gold_silver5k.jsonl.
- Automatically skips clips already cached in data/cache/audio_clips/.
- Streams sequentially up to max index with Audio(decode=False) to avoid torchcodec issues.
- Supports --dry-run for audit and manifest verification without downloading.
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Set

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-cache clean Silver audio clips from HuggingFace dataset")
    parser.add_argument("--num-samples", type=int, default=5000,
                        help="Number of clean Silver samples to select and cache (default: 5000)")
    parser.add_argument("--max-blanket-ratio", type=float, default=0.90,
                        help="Max allowable event duration / clip duration ratio (default: 0.90; >0.90 is excluded)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run metadata audit, selection, and manifest creation without downloading audio")
    parser.add_argument("--manifest-path", type=str, default="data/processed/train_gold_silver5k.jsonl",
                        help="Path to output combined Gold+Silver training manifest")
    return parser.parse_args()


def audit_and_select_silver(
    train_jsonl: Path,
    num_samples: int = 5000,
    max_blanket_ratio: float = 0.90,
) -> Tuple[List[Dict], Dict]:
    """
    Audit train_split.jsonl, apply noise-span filtering, and deterministically
    select the first num_samples clean Silver records.
    """
    total_silver = 0
    excluded_blanket = 0
    excluded_invalid = 0
    eligible_silver = []

    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("tier") != "silver":
                continue

            total_silver += 1
            clip_dur = float(rec.get("duration", 0.0))
            events = rec.get("events", [])

            # Check if any event spans more than max_blanket_ratio of clip duration
            is_blanket = False
            for ev in events:
                on = ev.get("onset")
                off = ev.get("offset")
                if on is not None and off is not None and clip_dur > 0:
                    try:
                        on_f, off_f = float(on), float(off)
                        if (off_f - on_f) / clip_dur > max_blanket_ratio:
                            is_blanket = True
                            break
                    except (ValueError, TypeError):
                        pass

            if is_blanket:
                excluded_blanket += 1
                continue

            # Ensure clip has at least one valid event interval
            has_valid_event = False
            for ev in events:
                on = ev.get("onset")
                off = ev.get("offset")
                if on is not None and off is not None:
                    try:
                        if float(off) > float(on) and float(on) >= 0.0:
                            has_valid_event = True
                            break
                    except (ValueError, TypeError):
                        pass

            if not has_valid_event:
                excluded_invalid += 1
                continue

            # Parse integer dataset index for deterministic ordering
            cid = rec["clip_id"]
            try:
                idx = int(cid.replace("train_", ""))
                eligible_silver.append((idx, rec))
            except ValueError:
                continue

    # Deterministic sort by Hugging Face dataset index
    eligible_silver.sort(key=lambda x: x[0])
    selected_silver = [item[1] for item in eligible_silver[:num_samples]]

    stats = {
        "total_silver_in_split": total_silver,
        "excluded_blanket_count": excluded_blanket,
        "excluded_blanket_pct": (excluded_blanket / total_silver * 100) if total_silver else 0.0,
        "excluded_invalid_count": excluded_invalid,
        "total_eligible_clean": len(eligible_silver),
        "selected_count": len(selected_silver),
    }

    return selected_silver, stats


def create_combined_manifest(
    train_jsonl: Path,
    selected_silver: List[Dict],
    cache_dir: Path,
    output_manifest: Path,
) -> Dict:
    """
    Create a verified training JSONL manifest containing all cached Gold clips
    plus the selected clean Silver clips.
    """
    cached_wavs = set(f.name for f in cache_dir.glob("*.wav") if f.stat().st_size > 0)

    gold_records = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("tier") == "gold":
                cid = rec["clip_id"]
                if f"{cid}.wav" in cached_wavs:
                    gold_records.append(rec)

    # Sort gold records deterministically by clip index
    gold_records.sort(key=lambda r: int(r["clip_id"].replace("train_", "")))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    all_combined = gold_records + selected_silver

    with open(output_manifest, "w", encoding="utf-8") as fout:
        for rec in all_combined:
            fout.write(json.dumps(rec) + "\n")

    return {
        "gold_count": len(gold_records),
        "silver_count": len(selected_silver),
        "total_manifest_count": len(all_combined),
        "output_path": str(output_manifest),
    }


def main():
    args = parse_args()

    print("=" * 75)
    print(f"VAANI TRACK 1: SILVER-TIER AUDIT & PRE-CACHING ({args.num_samples} CLIPS)")
    print("=" * 75)

    train_jsonl = PROJECT_ROOT / "data" / "processed" / "train_split.jsonl"
    cache_dir = PROJECT_ROOT / "data" / "cache" / "audio_clips"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT_ROOT / args.manifest_path

    # 1. Audit and deterministic Silver selection
    print(f"\n[Step 1/5] Auditing Silver annotations from {train_jsonl}...")
    selected_silver, stats = audit_and_select_silver(
        train_jsonl,
        num_samples=args.num_samples,
        max_blanket_ratio=args.max_blanket_ratio,
    )

    print(f"  - Total Silver clips in train_split    : {stats['total_silver_in_split']:,}")
    print(f"  - Excluded blanket events (>90% dur)   : {stats['excluded_blanket_count']:,} ({stats['excluded_blanket_pct']:.2f}%)")
    print(f"  - Excluded invalid / empty intervals   : {stats['excluded_invalid_count']:,}")
    print(f"  - Total clean, eligible Silver clips   : {stats['total_eligible_clean']:,}")
    print(f"  - Deterministically selected subset    : {stats['selected_count']:,} clips")

    if not selected_silver:
        raise RuntimeError("No clean Silver samples could be selected.")

    target_subset = []
    for rec in selected_silver:
        idx = int(rec["clip_id"].replace("train_", ""))
        target_subset.append((idx, rec["clip_id"]))

    min_idx = target_subset[0][0]
    max_idx = target_subset[-1][0]
    print(f"  - Target Silver index range            : [{min_idx} ('{target_subset[0][1]}') to {max_idx} ('{target_subset[-1][1]}')]")

    # 2. Check existing cache
    print(f"\n[Step 2/5] Inspecting local cache at {cache_dir}...")
    existing_files = set(f.name for f in cache_dir.glob("*.wav") if f.stat().st_size > 0)
    needed_indices = {}
    already_cached_count = 0

    for idx, cid in target_subset:
        fname = f"{cid}.wav"
        if fname in existing_files:
            already_cached_count += 1
        else:
            needed_indices[idx] = cid

    print(f"  - Already cached target Silver clips   : {already_cached_count}/{len(target_subset)}")
    print(f"  - Remaining to download and cache      : {len(needed_indices)}/{len(target_subset)}")

    # 3. Create combined training manifest
    print(f"\n[Step 3/5] Generating combined training manifest...")
    manifest_stats = create_combined_manifest(train_jsonl, selected_silver, cache_dir, manifest_path)
    print(f"  - Cached Gold clips included           : {manifest_stats['gold_count']:,}")
    print(f"  - Clean Silver clips included          : {manifest_stats['silver_count']:,}")
    print(f"  - Total unified training samples       : {manifest_stats['total_manifest_count']:,}")
    print(f"  - Written to                           : {manifest_stats['output_path']}")

    # 4. Handle streaming / downloading
    if args.dry_run:
        print("\n[Step 4/5] DRY-RUN mode active: Skipping Hugging Face download.")
        print("  - Metadata audit and manifest creation completed successfully.")
        return

    if not needed_indices:
        print("\n[Step 4/5] All target Silver clips are already cached! No streaming needed.")
    else:
        load_dotenv(PROJECT_ROOT / ".env")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN not found in .env or environment variables.")

        print(f"\n[Step 4/5] Initializing Hugging Face streaming dataset (decode=False)...")
        t_init = time.time()
        ds = load_dataset(
            "ARTPARK-IISc/Vaani-Noise-Event-Dataset",
            token=token,
            streaming=True,
        )
        train_stream = ds["train"].cast_column("audio", Audio(decode=False))
        print(f"  - Stream initialized in {time.time() - t_init:.2f}s")

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

                if cached_in_run % 50 == 0 or (already_cached_count + cached_in_run) == len(target_subset):
                    elapsed = time.time() - t_start
                    total_done = already_cached_count + cached_in_run
                    rate = cached_in_run / max(elapsed, 1e-3)
                    print(
                        f"[{total_done:5d}/{len(target_subset)}] Cached {cid} (idx {idx:5d}) | "
                        f"Run: {cached_in_run} clips in {elapsed:.1f}s ({rate:.1f} clips/s)",
                        flush=True,
                    )

            if (already_cached_count + cached_in_run) >= len(target_subset):
                print(f"\nAll {len(target_subset)} target Silver clips have been cached! Stopping stream early.")
                break

        total_elapsed = time.time() - t_start
        print(f"\nStreaming pass finished in {total_elapsed:.1f}s. Newly cached: {cached_in_run} clips.")

    # 5. Verification
    print("\n" + "=" * 75)
    print(f"[Step 5/5] VERIFICATION OF CACHED SILVER SAMPLES")
    print("=" * 75)

    missing = []
    target_sizes = []
    for idx, cid in target_subset:
        p = cache_dir / f"{cid}.wav"
        if not p.exists() or p.stat().st_size == 0:
            missing.append(cid)
        else:
            target_sizes.append(p.stat().st_size)

    if missing:
        print(f"WARNING: {len(missing)} target clips are not yet cached on disk.")
        print(f"Missing sample IDs: {missing[:10]}")
    else:
        print(f"SUCCESS: All {len(target_subset)} target Silver WAV files verified in {cache_dir}!")

    total_target_bytes = sum(target_sizes)
    all_cached_files = list(cache_dir.glob("*.wav"))
    total_all_bytes = sum(f.stat().st_size for f in all_cached_files)

    print(f"  - Target Silver clips verified : {len(target_sizes)}/{len(target_subset)}")
    print(f"  - Target Silver audio size     : {total_target_bytes / (1024 * 1024):.2f} MB")
    print(f"  - Total cache files            : {len(all_cached_files)} WAVs")
    print(f"  - Total cache directory size   : {total_all_bytes / (1024 * 1024):.2f} MB")
    print("=" * 75)


if __name__ == "__main__":
    main()
