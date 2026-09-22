"""
dataset.py
PyTorch Dataset wrapper for Vaani Noise Event Detection.
Integrates with audio_loader.py for lazy loading and local caching.

Usage:
    from src.dataset import VaaniNoiseDataset, collate_fn_vaani
    from torch.utils.data import DataLoader

    train_dataset = VaaniNoiseDataset(
        jsonl_path="data/processed/train_split.jsonl",
        cache_dir="data/cache/audio_clips"
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                               collate_fn=collate_fn_vaani)
"""

import json
from pathlib import Path
from typing import Optional, List, Dict

import torch
import numpy as np
from torch.utils.data import Dataset

from .audio_loader import AudioLoader


class VaaniNoiseDataset(Dataset):
    """
    PyTorch Dataset for Vaani Noise Event Detection.

    - Lazy audio loading with caching (via AudioLoader)
    - Supports Gold/Silver/Bronze annotation tiers
    - Returns audio waveform + event annotations + metadata
    """

    def __init__(
        self,
        jsonl_path: str,
        cache_dir: str = "data/cache/audio_clips",
        target_sr: int = 16000,
        max_duration: Optional[float] = None,
        tier_filter: Optional[List[str]] = None,
        transform=None,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.cache_dir = cache_dir
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.tier_filter = tier_filter
        self.transform = transform

        self.audio_loader = AudioLoader(cache_dir=cache_dir, target_sr=target_sr)
        self.samples = self._load_metadata()

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")
        if tier_filter:
            print(f"Filtered to tiers: {tier_filter}")

    def _load_metadata(self) -> List[Dict]:
        samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if self.tier_filter is not None and record["tier"] not in self.tier_filter:
                    continue
                samples.append(record)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        clip_id = sample["clip_id"]

        try:
            audio, sr = self.audio_loader.load_audio(clip_id)
        except Exception as e:
            print(f"Error loading audio for {clip_id}: {e}")
            audio = np.zeros(int(self.target_sr * (self.max_duration or 10)), dtype=np.float32)
            sr = self.target_sr

        if self.max_duration is not None:
            target_length = int(self.max_duration * sr)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            elif len(audio) > target_length:
                audio = audio[:target_length]

        if self.transform is not None:
            audio = self.transform(audio)

        audio_tensor = torch.from_numpy(audio).float()

        return {
            "audio": audio_tensor,
            "sample_rate": sr,
            "events": sample.get("events", []),
            "clip_id": clip_id,
            "tier": sample["tier"],
            "language": sample.get("language", "unknown"),
            "duration": sample.get("duration", len(audio) / sr),
            "transcript": sample.get("transcript", ""),
            "noise_tags": sample.get("noise_tags", []),
        }

    def get_tier_distribution(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(s["tier"] for s in self.samples))

    def get_language_distribution(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(s.get("language", "unknown") for s in self.samples))

    def get_cache_stats(self) -> Dict:
        return self.audio_loader.get_cache_stats()


def collate_fn_vaani(batch: List[Dict]) -> Dict:
    max_length = max(sample["audio"].shape[0] for sample in batch)
    batch_size = len(batch)

    audio_padded = torch.zeros(batch_size, max_length)
    for i, sample in enumerate(batch):
        audio_len = sample["audio"].shape[0]
        audio_padded[i, :audio_len] = sample["audio"]

    return {
        "audio": audio_padded,
        "sample_rate": batch[0]["sample_rate"],
        "events": [s["events"] for s in batch],
        "clip_ids": [s["clip_id"] for s in batch],
        "tiers": [s["tier"] for s in batch],
        "languages": [s["language"] for s in batch],
        "durations": torch.tensor([s["duration"] for s in batch]),
        "transcripts": [s["transcript"] for s in batch],
        "noise_tags": [s["noise_tags"] for s in batch],
    }
