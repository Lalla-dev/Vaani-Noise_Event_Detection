# Audio Loading System Guide

## Overview

The audio loading system solves the **disk space problem** by implementing **lazy loading with local caching**. Instead of downloading all 90k+ audio files upfront, it fetches audio clips on-demand from HuggingFace and caches them locally for reuse.

---

## Architecture

```
┌─────────────────────┐
│  PyTorch Dataset    │
│   (dataset.py)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   AudioLoader       │
│  (audio_loader.py)  │
└──────────┬──────────┘
           │
      ┌────┴────┐
      │         │
      ▼         ▼
┌─────────┐ ┌──────────────┐
│  Cache  │ │ HuggingFace  │
│  (WAV)  │ │  Streaming   │
└─────────┘ └──────────────┘
```

---

## Components

### 1. **AudioLoader** (`src/audio_loader.py`)
Core utility that handles:
- ✅ Lazy loading from HuggingFace
- ✅ Local caching as `.wav` files
- ✅ Clip ID → HF index mapping
- ✅ Automatic resampling to target SR
- ✅ Cache management utilities

### 2. **VaaniNoiseDataset** (`src/dataset.py`)
PyTorch Dataset that:
- ✅ Integrates AudioLoader
- ✅ Loads metadata from JSONL splits
- ✅ Returns audio + annotations
- ✅ Supports tier filtering
- ✅ Custom collate function for batching

---

## Usage Examples

### Quick Start: Load a Single Clip

```python
from src.audio_loader import AudioLoader

# Initialize loader
loader = AudioLoader(cache_dir="data/cache/audio_clips", target_sr=16000)

# Load audio (fetches from HF if not cached)
audio, sr = loader.load_audio("train_000042")
print(f"Audio shape: {audio.shape}, SR: {sr}")
# Output: Audio shape: (72800,), SR: 16000
```

### Using with PyTorch Dataset

```python
from src.dataset import VaaniNoiseDataset, collate_fn_vaani
from torch.utils.data import DataLoader

# Create dataset
train_dataset = VaaniNoiseDataset(
    jsonl_path="data/processed/train_split.jsonl",
    cache_dir="data/cache/audio_clips",
    target_sr=16000,
    max_duration=10.0,  # Pad/truncate to 10 seconds
    tier_filter=["gold", "silver"]  # Only use Gold & Silver tiers
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn_vaani,
    num_workers=2  # Parallel loading
)

# Training loop
for batch in train_loader:
    audio = batch['audio']          # Shape: (batch_size, max_length)
    events = batch['events']        # List of event lists
    clip_ids = batch['clip_ids']    # List of clip IDs
    
    # Your training code here
    # ...
```

### Batch Loading Multiple Clips

```python
from src.audio_loader import AudioLoader

loader = AudioLoader()

clip_ids = ["train_000001", "train_000002", "train_000003"]
results = loader.batch_load(clip_ids, show_progress=True)

for clip_id, (audio, sr) in results.items():
    print(f"{clip_id}: {audio.shape}")
```

### Cache Management

```python
from src.audio_loader import AudioLoader

loader = AudioLoader()

# Get cache statistics
stats = loader.get_cache_stats()
print(f"Cached files: {stats['num_cached']}")
print(f"Total size: {stats['total_size_mb']} MB")

# Clear specific clip from cache
loader.clear_cache("train_000042")

# Clear entire cache
loader.clear_cache()
```

---

## How It Works

### Clip ID Mapping

The system uses sequential `clip_id` (e.g., `train_000042`) to map back to HuggingFace dataset indices:

```
train_000000 → HF index 0
train_000001 → HF index 1
train_000042 → HF index 42
...
```

This mapping is **deterministic** and requires no additional lookup tables since the JSONL files were generated sequentially from the HF dataset.

### Caching Strategy

1. **First access:** Fetch from HF → Save to `data/cache/audio_clips/train_XXXXXX.wav`
2. **Subsequent access:** Load directly from local cache (instant)
3. **Cache location:** `data/cache/audio_clips/` (gitignored)

### Storage Estimates

- **Single clip:** ~100-500 KB (16kHz WAV)
- **1000 clips:** ~100-500 MB
- **All 90k clips:** ~9-45 GB (only if you load everything)

**Strategy:** Only cache the clips you actually use during training/validation.

---

## Configuration Options

### AudioLoader Parameters

```python
AudioLoader(
    cache_dir="data/cache/audio_clips",  # Where to store cached audio
    dataset_name="ARTPARK-IISc/Vaani-Noise-Event-Dataset",  # HF dataset
    use_streaming=True,  # Stream from HF (recommended)
    target_sr=16000      # Target sampling rate (None = keep original)
)
```

### VaaniNoiseDataset Parameters

```python
VaaniNoiseDataset(
    jsonl_path="data/processed/train_split.jsonl",  # JSONL split file
    cache_dir="data/cache/audio_clips",             # Audio cache directory
    target_sr=16000,                                 # Target sampling rate
    max_duration=10.0,                               # Pad/truncate to N seconds
    tier_filter=["gold", "silver"],                  # Filter by annotation tier
    transform=None                                   # Optional audio augmentation
)
```

---

## Integration with Training Pipeline

### Recommended Workflow

```python
# 1. Create datasets
train_dataset = VaaniNoiseDataset(
    "data/processed/train_split.jsonl",
    tier_filter=["gold", "silver"]  # Start with Gold+Silver only
)

val_dataset = VaaniNoiseDataset(
    "data/processed/val_split.jsonl",
    tier_filter=["gold", "silver"]
)

# 2. Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn_vaani,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn_vaani,
    num_workers=2
)

# 3. Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        audio = batch['audio']  # (batch_size, max_length)
        events = batch['events']
        
        # Forward pass, loss, backward, optimize
        # ...
```

---

## Testing

### Test AudioLoader

```bash
python src/audio_loader.py
```

This will:
1. Load a few test clips from HF
2. Cache them locally
3. Display cache statistics

### Test PyTorch Dataset

```bash
python src/dataset.py
```

This will:
1. Load the training split
2. Show tier/language distributions
3. Load a sample and display info
4. Test DataLoader batching
5. Display cache stats

---

## Troubleshooting

### Issue: "HF_TOKEN not found"
**Solution:** Copy `.env.example` to `.env` and add your HuggingFace token:
```bash
cp .env.example .env
# Edit .env and add: HF_TOKEN=your_token_here
```

### Issue: Slow first-time loading
**Expected behavior:** First access fetches from HF (slow). Subsequent access uses cache (fast).

### Issue: Corrupted cache files
**Solution:** Clear cache and reload:
```python
loader.clear_cache("train_000042")  # Clear specific clip
loader.clear_cache()                # Or clear all
```

### Issue: Out of disk space
**Solution:** Monitor cache size and clear old clips:
```python
stats = loader.get_cache_stats()
if stats['total_size_mb'] > 5000:  # >5GB
    loader.clear_cache()  # Clear and start fresh
```

---

## Next Steps for Team

1. **Shubham:** Integrate `VaaniNoiseDataset` into your baseline model training
2. **Ankit & Rithish:** Use this for advanced model experiments
3. **Soubhik:** Use for Bronze-tier weak supervision experiments
4. **Everyone:** Monitor cache size, especially on laptops with limited storage

---

## Performance Tips

1. **Use `num_workers > 0`** in DataLoader for parallel loading
2. **Cache frequently used clips** during development
3. **Filter by tier** to reduce dataset size for quick experiments
4. **Set `max_duration`** to pad/truncate for consistent batch shapes
5. **Periodic cache cleanup** to manage disk space

---

## Questions?

Ask in the team channel or check the inline documentation in:
- `src/audio_loader.py`
- `src/dataset.py`
