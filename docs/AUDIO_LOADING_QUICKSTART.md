# 🎵 Audio Loading System - Quick Start

## TL;DR
**Problem:** Downloading all 90k+ audio files = 💥 disk space explosion  
**Solution:** Lazy loading + local caching = 💚 fetch only what you need

---

## 🚀 Quick Start (3 Steps)

### 1️⃣ Test the System
```bash
python test_audio_system.py
```

This will:
- ✅ Load a few test clips from HuggingFace
- ✅ Cache them locally in `data/cache/audio_clips/`
- ✅ Verify PyTorch Dataset integration works
- ✅ Confirm cache reuse is working

### 2️⃣ Use in Training (Simple)
```python
from src.dataset import VaaniNoiseDataset, collate_fn_vaani
from torch.utils.data import DataLoader

# Create dataset
train_dataset = VaaniNoiseDataset(
    jsonl_path="data/processed/train_split.jsonl",
    cache_dir="data/cache/audio_clips",
    target_sr=16000,
    max_duration=10.0  # Pad/truncate to 10 seconds
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn_vaani,
    num_workers=2
)

# Training loop
for batch in train_loader:
    audio = batch['audio']      # Shape: (16, 160000) at 16kHz
    events = batch['events']    # List of event annotations
    clip_ids = batch['clip_ids']
    # Your training code here...
```

### 3️⃣ Monitor Cache Size
```python
from src.audio_loader import AudioLoader

loader = AudioLoader()
stats = loader.get_cache_stats()

print(f"Cached: {stats['num_cached']} files, {stats['total_size_mb']} MB")

# Clear cache if needed
# loader.clear_cache()  # Clear all
# loader.clear_cache("train_000042")  # Clear specific clip
```

---

## 📊 What You Get

Each batch contains:
```python
{
    'audio': torch.Tensor,           # (batch_size, max_length)
    'sample_rate': int,              # e.g., 16000
    'events': List[List[Dict]],      # Event annotations per sample
    'clip_ids': List[str],           # e.g., ["train_000001", ...]
    'tiers': List[str],              # ["gold", "silver", "bronze"]
    'languages': List[str],          # ["Hindi", "Telugu", ...]
    'durations': torch.Tensor,       # Original durations in seconds
    'transcripts': List[str],        # Transcripts with noise tags
    'noise_tags': List[List[str]]    # For bronze tier
}
```

Event format (Gold/Silver):
```python
{
    'onset': 1.845,         # Start time in seconds
    'offset': 3.902,        # End time in seconds
    'category': 'animal'    # Noise category
}
```

---

## 🎯 Team-Specific Usage

### Shubham (Baseline Model)
```python
# Use Gold + Silver tiers only for baseline
dataset = VaaniNoiseDataset(
    "data/processed/train_split.jsonl",
    tier_filter=["gold", "silver"]  # Ignore bronze for now
)
```

### Ankit & Rithish (Advanced Modeling)
```python
# Experiment with pretrained encoders
# Audio is already resampled to 16kHz by default
dataset = VaaniNoiseDataset(
    "data/processed/train_split.jsonl",
    target_sr=16000,  # Standard for most pretrained models
    max_duration=10.0  # Adjust based on model input requirements
)
```

### Soubhik (Bronze Weak Supervision)
```python
# Bronze tier has tags but no timestamps
bronze_dataset = VaaniNoiseDataset(
    "data/processed/train_split.jsonl",
    tier_filter=["bronze"]
)

# Access noise tags instead of events
for sample in bronze_dataset:
    tags = sample['noise_tags']  # List of tag names (no timestamps)
    # Your weak supervision logic here...
```

---

## 💡 Pro Tips

1. **Start small:** Test with a few clips before full training
2. **Use tier filtering:** Focus on Gold/Silver for initial experiments
3. **Monitor cache:** Check `get_cache_stats()` periodically
4. **Parallel loading:** Set `num_workers > 0` in DataLoader
5. **Cache is gitignored:** Won't bloat your repo

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "HF_TOKEN not found" | Copy `.env.example` to `.env` and add your token |
| Slow first load | Expected! Fetching from HF. Subsequent loads are fast (cache) |
| Out of disk space | Run `loader.clear_cache()` to free up space |
| Import errors | Make sure you're in the repo root: `cd Vaani-Noise_Event_Detection-main` |

---

## 📚 More Details

See full documentation: [`docs/audio_loading_guide.md`](docs/audio_loading_guide.md)

---

## ✅ Checklist Before Training

- [ ] Run `python test_audio_system.py` - all tests pass
- [ ] `.env` file exists with your HF_TOKEN
- [ ] `data/processed/train_split.jsonl` exists (run `python src/split_data.py` if not)
- [ ] `data/cache/audio_clips/` directory is gitignored
- [ ] Understand cache size will grow with usage (~100-500KB per clip)

---

**Questions?** Check inline docs in `src/audio_loader.py` and `src/dataset.py`

Happy training! 🚀
