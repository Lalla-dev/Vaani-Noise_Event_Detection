"""
verify_setup.py
Verify the audio loading system setup (no heavy dependencies needed for this check)
"""

import json
import sys
from pathlib import Path

print("=" * 70)
print("AUDIO LOADING SYSTEM - SETUP VERIFICATION")
print("=" * 70)

all_good = True

# Check 1: Files exist
print("\n✓ Step 1: Checking files...")
files_to_check = {
    "src/audio_loader.py": "Audio loader utility",
    "src/dataset.py": "PyTorch Dataset wrapper",
    "data/processed/unified.jsonl": "Unified dataset",
    "data/processed/train_split.jsonl": "Training split",
    "data/processed/val_split.jsonl": "Validation split",
    ".env.example": "Environment template"
}

for file_path, description in files_to_check.items():
    if Path(file_path).exists():
        print(f"  ✓ {file_path:<40} ({description})")
    else:
        print(f"  ✗ {file_path:<40} MISSING!")
        all_good = False

# Check 2: .env file
print("\n✓ Step 2: Checking environment configuration...")
env_path = Path(".env")
if env_path.exists():
    print(f"  ✓ .env file exists")
    # Try to check if HF_TOKEN is set
    with open(".env", 'r') as f:
        content = f.read()
        if "HF_TOKEN=" in content and not "HF_TOKEN=your_token_here" in content:
            print(f"  ✓ HF_TOKEN appears to be configured")
        else:
            print(f"  ⚠  HF_TOKEN may not be set properly")
            print(f"     Please ensure .env contains: HF_TOKEN=your_actual_token")
else:
    print(f"  ✗ .env file NOT FOUND")
    print(f"     Create it by: copy .env.example to .env and add your HF token")
    all_good = False

# Check 3: Data structure
print("\n✓ Step 3: Verifying data structure...")

jsonl_path = Path("data/processed/train_split.jsonl")
if jsonl_path.exists():
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            # Read first sample
            first_line = f.readline()
            sample = json.loads(first_line)
            
        print(f"  ✓ JSONL file is valid JSON")
        print(f"  ✓ Sample clip_id: {sample['clip_id']}")
        print(f"  ✓ Sample tier: {sample['tier']}")
        print(f"  ✓ Sample language: {sample.get('language', 'unknown')}")
        print(f"  ✓ Sample duration: {sample.get('duration', 'N/A')}s")
        print(f"  ✓ Sample events: {len(sample.get('events', []))} events")
        
        # Validate clip_id format
        clip_id = sample['clip_id']
        if clip_id.startswith("train_") and clip_id[6:].isdigit():
            index = int(clip_id[6:])
            print(f"  ✓ Clip ID format is valid (maps to index: {index})")
        else:
            print(f"  ✗ Clip ID format seems invalid: {clip_id}")
            all_good = False
            
    except Exception as e:
        print(f"  ✗ Error reading JSONL: {e}")
        all_good = False
else:
    print(f"  ✗ train_split.jsonl not found")
    all_good = False

# Check 4: Code structure
print("\n✓ Step 4: Checking audio_loader.py structure...")

audio_loader_path = Path("src/audio_loader.py")
if audio_loader_path.exists():
    with open(audio_loader_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Check for key components
    checks = [
        ("class AudioLoader", "AudioLoader class"),
        ("def load_audio", "load_audio method"),
        ("def _fetch_from_hf", "HuggingFace fetching"),
        ("def _get_cache_path", "Cache path generation"),
        ("def _save_to_cache", "Cache saving"),
        ("def get_cache_stats", "Cache statistics"),
    ]
    
    for pattern, description in checks:
        if pattern in code:
            print(f"  ✓ {description:<30} found")
        else:
            print(f"  ✗ {description:<30} MISSING!")
            all_good = False

# Check 5: Test mapping logic
print("\n✓ Step 5: Testing clip_id → index mapping logic...")

def extract_index(clip_id):
    """Test the mapping function"""
    try:
        return int(clip_id.replace("train_", ""))
    except:
        return None

test_cases = [
    ("train_000000", 0),
    ("train_000001", 1),
    ("train_000042", 42),
    ("train_012345", 12345),
]

for clip_id, expected in test_cases:
    result = extract_index(clip_id)
    if result == expected:
        print(f"  ✓ {clip_id} → index {result}")
    else:
        print(f"  ✗ {clip_id} → index {result} (expected {expected})")
        all_good = False

# Check 6: Cache directory
print("\n✓ Step 6: Checking cache directory...")
cache_dir = Path("data/cache/audio_clips")
if cache_dir.exists():
    cached_files = list(cache_dir.glob("*.wav"))
    print(f"  ✓ Cache directory exists: {cache_dir}")
    print(f"  ℹ  Currently cached: {len(cached_files)} files")
    if len(cached_files) > 0:
        total_size_mb = sum(f.stat().st_size for f in cached_files) / (1024 * 1024)
        print(f"  ℹ  Cache size: {total_size_mb:.2f} MB")
else:
    print(f"  ℹ  Cache directory doesn't exist yet (will be created on first use)")

# Check 7: .gitignore
print("\n✓ Step 7: Checking .gitignore...")
gitignore_path = Path(".gitignore")
if gitignore_path.exists():
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    if "data/cache/" in gitignore_content:
        print(f"  ✓ data/cache/ is gitignored (won't bloat repo)")
    else:
        print(f"  ⚠  data/cache/ may not be gitignored")
        print(f"     Consider adding 'data/cache/' to .gitignore")
    
    if ".env" in gitignore_content:
        print(f"  ✓ .env is gitignored (tokens are safe)")
    else:
        print(f"  ⚠  .env may not be gitignored (security risk!)")

# Summary
print("\n" + "=" * 70)
if all_good:
    print("✅ SETUP VERIFICATION PASSED!")
    print("=" * 70)
    print("\nEverything looks good! Next steps:")
    print()
    print("1. Install dependencies (if not already done):")
    print("   pip install -r requirements.txt")
    print()
    print("2. Make sure .env has your HuggingFace token:")
    print("   HF_TOKEN=your_actual_token_here")
    print()
    print("3. Test the audio loading:")
    print("   python quick_test.py")
    print()
    print("   This will download 1 audio clip from HuggingFace to verify")
    print("   the system works correctly.")
    print()
    print("4. Start using in your training code:")
    print("   from src.dataset import VaaniNoiseDataset")
    print()
else:
    print("⚠️  SETUP VERIFICATION FOUND ISSUES")
    print("=" * 70)
    print("\nPlease fix the issues marked with ✗ above before proceeding.")
    print()
    sys.exit(1)
