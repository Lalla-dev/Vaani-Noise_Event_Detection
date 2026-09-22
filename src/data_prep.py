"""
data_prep.py
Fetch the Vaani-Noise-Event-Dataset from Hugging Face and unify the three
annotation tiers (Gold / Silver / Bronze) into a single clean format that
downstream training code (dataset.py) can consume.

Usage:
    python src/data_prep.py --out data/processed/unified.jsonl

Requires a .env file (copy .env.example -> .env) with:
    HF_TOKEN=your_real_token
"""

import argparse
import json
import os
import re
from pathlib import Path

from datasets import Audio  
from datasets import load_dataset
from dotenv import load_dotenv

DATASET_NAME = "ARTPARK-IISc/Vaani-Noise-Event-Dataset"

# Bronze-tier transcripts embed tags inline like:
#   <noise> ... <horn> </horn> ... </noise>
# We extract the tag names only (no timestamps available at this tier).
TAG_PATTERN = re.compile(r"<(\w+)>\s*</\1>")

def load_hf_dataset():
    """Load the dataset dictionary using streaming."""
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Copy .env.example to .env and add your token."
        )
    return load_dataset(DATASET_NAME, token=token, streaming=True)


def parse_timestamped_events(raw_field):
    """
    Gold/Silver samples store events as a JSON-ish list, e.g.:
    [
      {"category": "vehicle_traffic", "tag": "<horn>", "start": "2.714", "end": "3.761"},
      {"category": "human_non_speech", "tag": "[breathing]", "start": "4.938", "end": "5.410"},
      {"Verification_status": "Verified"}
    ]
    Returns a list of {"onset": float, "offset": float, "category": str}.
    NOTE: verify the exact field name/shape against a real sample before
    trusting this in production -- print sample['NoiseSubCategoryTimeStamp']
    for a few rows first.
    """
    if not raw_field:
        return []

    try:
        entries = raw_field if isinstance(raw_field, list) else json.loads(raw_field)
    except (json.JSONDecodeError, TypeError):
        return []

    events = []
    for entry in entries:
        if "start" in entry and "end" in entry:
            try:
                events.append(
                    {
                        "onset": float(entry["start"]),
                        "offset": float(entry["end"]),
                        "category": entry.get("category", "unknown"),
                    }
                )
            except (ValueError, TypeError):
                continue
    return events


def parse_bronze_tags(transcript):
    """Bronze tier has no timestamps -- just pull out the tag names present."""
    if not transcript:
        return []
    return TAG_PATTERN.findall(transcript)


def unify_sample(sample, idx):
    quality = (sample.get("annotationQuality") or "").lower()

    if "no_timestamps" in quality:
        tier = "bronze"
    elif "unverified" in quality:
        tier = "silver"
    elif "verified" in quality:
        tier = "gold"
    else:
        tier = "unknown"

    record = {
        "clip_id": f"train_{idx:06d}",
        "tier": tier,
        "language": sample.get("language"),
        "state": sample.get("state"),
        "district": sample.get("district"),
        "duration": sample.get("duration"),
        "transcript": sample.get("transcript"),
        "events": [],
        "noise_tags": [],
    }

    if tier in ("gold", "silver"):
        record["events"] = parse_timestamped_events(sample.get("NoiseSubCategoryTimeStamp"))
    elif tier == "bronze":
        record["noise_tags"] = parse_bronze_tags(sample.get("transcript"))
    else:
        record["events"] = parse_timestamped_events(sample.get("NoiseSubCategoryTimeStamp"))
        record["noise_tags"] = parse_bronze_tags(sample.get("transcript"))

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/processed/unified.jsonl")
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional cap for quick local testing"
    )
    args = parser.parse_args()

    dataset = load_hf_dataset()
    
    train_split = dataset["train"].cast_column("audio", Audio(decode=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tier_counts = {"gold": 0, "silver": 0, "bronze": 0, "unknown": 0}
    total_records = 0

    print("Processing started (Streaming text-only from Hugging Face)...")

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(train_split):
            if args.limit is not None and i >= args.limit:
                break
                
            record = unify_sample(sample, i)
            tier_counts[record["tier"]] = tier_counts.get(record["tier"], 0) + 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_records += 1

            if total_records % 1000 == 0:
                print(f"Processed {total_records} samples...")

    print(f"\nSuccess! Wrote {total_records} records to {out_path}")
    print("Tier breakdown:", tier_counts)


if __name__ == "__main__":
    main()