"""
test_audio_system.py
Quick test script to verify the audio loading system works correctly.

Usage:
    python test_audio_system.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_audio_loader():
    """Test basic AudioLoader functionality."""
    print("=" * 60)
    print("TEST 1: AudioLoader Basic Functionality")
    print("=" * 60)
    
    from src.audio_loader import AudioLoader
    
    # Initialize loader
    loader = AudioLoader(
        cache_dir="data/cache/audio_clips",
        target_sr=16000
    )
    
    # Test loading a few clips
    test_clips = ["train_000001", "train_000002", "train_000010"]
    
    print(f"\nLoading {len(test_clips)} test clips...\n")
    
    for clip_id in test_clips:
        try:
            audio, sr = loader.load_audio(clip_id)
            duration = len(audio) / sr
            print(f"✓ {clip_id}: shape={audio.shape}, sr={sr}, duration={duration:.2f}s")
        except Exception as e:
            print(f"✗ {clip_id}: ERROR - {e}")
            return False
    
    # Check cache stats
    print("\n--- Cache Statistics ---")
    stats = loader.get_cache_stats()
    print(f"  Cached files: {stats['num_cached']}")
    print(f"  Total size: {stats['total_size_mb']} MB")
    print(f"  Cache directory: {stats['cache_dir']}")
    
    print("\n✓ AudioLoader test PASSED\n")
    return True


def test_dataset():
    """Test VaaniNoiseDataset functionality."""
    print("=" * 60)
    print("TEST 2: VaaniNoiseDataset Functionality")
    print("=" * 60)
    
    from src.dataset import VaaniNoiseDataset, collate_fn_vaani
    from torch.utils.data import DataLoader
    
    # Check if splits exist
    train_path = Path("data/processed/train_split.jsonl")
    if not train_path.exists():
        print(f"✗ Train split not found at {train_path}")
        print("  Please run: python src/split_data.py")
        return False
    
    # Create dataset
    print(f"\nLoading dataset from {train_path}...\n")
    
    try:
        dataset = VaaniNoiseDataset(
            jsonl_path=str(train_path),
            cache_dir="data/cache/audio_clips",
            target_sr=16000,
            max_duration=10.0,
            tier_filter=["gold", "silver"]  # Test with Gold+Silver only
        )
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Show distributions
    print("\n--- Dataset Statistics ---")
    print(f"  Tier distribution: {dataset.get_tier_distribution()}")
    lang_dist = dataset.get_language_distribution()
    top_5_langs = dict(list(lang_dist.items())[:5])
    print(f"  Top 5 languages: {top_5_langs}")
    
    # Test loading a single sample
    print("\n--- Testing Single Sample ---")
    try:
        sample = dataset[0]
        print(f"  Clip ID: {sample['clip_id']}")
        print(f"  Audio shape: {sample['audio'].shape}")
        print(f"  Sample rate: {sample['sample_rate']}")
        print(f"  Tier: {sample['tier']}")
        print(f"  Language: {sample['language']}")
        print(f"  Duration: {sample['duration']:.2f}s")
        print(f"  Events: {len(sample['events'])} events")
        print(f"  ✓ Single sample load PASSED")
    except Exception as e:
        print(f"  ✗ Failed to load sample: {e}")
        return False
    
    # Test DataLoader
    print("\n--- Testing DataLoader ---")
    try:
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn_vaani,
            num_workers=0  # Use 0 for testing to avoid multiprocessing issues
        )
        
        batch = next(iter(loader))
        print(f"  Batch audio shape: {batch['audio'].shape}")
        print(f"  Batch size: {len(batch['clip_ids'])}")
        print(f"  Batch clip_ids: {batch['clip_ids']}")
        print(f"  ✓ DataLoader test PASSED")
    except Exception as e:
        print(f"  ✗ DataLoader test FAILED: {e}")
        return False
    
    print("\n✓ VaaniNoiseDataset test PASSED\n")
    return True


def test_cache_reuse():
    """Test that cached files are reused."""
    print("=" * 60)
    print("TEST 3: Cache Reuse")
    print("=" * 60)
    
    from src.audio_loader import AudioLoader
    import time
    
    loader = AudioLoader(cache_dir="data/cache/audio_clips")
    test_clip = "train_000001"
    
    # First load (might be from cache if previous tests ran)
    print(f"\nLoading {test_clip} (1st time)...")
    start = time.time()
    audio1, sr1 = loader.load_audio(test_clip)
    time1 = time.time() - start
    print(f"  Time: {time1:.3f}s")
    
    # Second load (should be from cache)
    print(f"\nLoading {test_clip} (2nd time - should be from cache)...")
    start = time.time()
    audio2, sr2 = loader.load_audio(test_clip)
    time2 = time.time() - start
    print(f"  Time: {time2:.3f}s")
    
    # Verify both loads return same data
    import numpy as np
    if np.array_equal(audio1, audio2) and sr1 == sr2:
        print(f"\n✓ Cache reuse PASSED (2nd load was {time1/max(time2, 0.001):.1f}x faster)")
    else:
        print("\n✗ Cache reuse FAILED - data mismatch")
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VAANI AUDIO LOADING SYSTEM - TEST SUITE")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: AudioLoader
    try:
        results.append(("AudioLoader", test_audio_loader()))
    except Exception as e:
        print(f"\n✗ AudioLoader test CRASHED: {e}\n")
        results.append(("AudioLoader", False))
    
    # Test 2: Dataset
    try:
        results.append(("Dataset", test_dataset()))
    except Exception as e:
        print(f"\n✗ Dataset test CRASHED: {e}\n")
        results.append(("Dataset", False))
    
    # Test 3: Cache reuse
    try:
        results.append(("Cache Reuse", test_cache_reuse()))
    except Exception as e:
        print(f"\n✗ Cache reuse test CRASHED: {e}\n")
        results.append(("Cache Reuse", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Audio loading system is ready to use.\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
