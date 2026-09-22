"""
audio_loader.py
Lazy audio loading utility with local caching for the Vaani-Noise-Event-Dataset.

Solves the disk space problem by:
1. Streaming only the required audio sample from HuggingFace on-demand
2. Caching downloaded files locally for reuse
3. Mapping clip_id back to the original HF dataset index (verified stable/sequential)
4. Uses .skip() for fast dataset access instead of iterating from index 0

IMPORTANT: Newer versions of the `datasets` library default to torchcodec for
audio decoding as soon as the audio column's schema says Audio(decode=True) --
which is the dataset's own default, so it happens even if we never call
.cast_column() ourselves. To avoid needing torchcodec/FFmpeg DLLs on Windows,
we explicitly cast the audio column to Audio(decode=False) so `datasets`
hands us raw encoded bytes instead of trying to decode anything itself, and
we decode those bytes ourselves with `soundfile` (via BytesIO). Do not remove
this cast or decode the audio column automatically without confirming
torchcodec/FFmpeg is properly installed on every team member's machine.

Usage:
    from src.audio_loader import AudioLoader

    loader = AudioLoader(cache_dir="data/cache/audio_clips")
    audio_array, sr = loader.load_audio("train_000042")
"""

import io
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import torchaudio
from datasets import Audio
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm


class AudioLoader:
    """
    On-demand audio loader with local caching for the Vaani dataset.

    Features:
    - Lazy loading: only downloads audio when requested
    - Local caching: saves .wav files to avoid re-downloading
    - Stable mapping: uses clip_id (train_XXXXXX) to fetch from HF dataset
      (verified against unified.jsonl: sequential, no shuffling)
    - Fast access: uses .skip(target_index) instead of iterating from 0
    - No torchcodec dependency: uses the datasets library's default audio
      decoding (soundfile-backed), which works out of the box on Windows/Linux/Mac
    """

    def __init__(
        self,
        cache_dir: str = "data/cache/audio_clips",
        dataset_name: str = "ARTPARK-IISc/Vaani-Noise-Event-Dataset",
        use_streaming: bool = True,
        target_sr: Optional[int] = 16000,
    ):
        """
        Args:
            cache_dir: Directory to store cached audio files
            dataset_name: HuggingFace dataset identifier
            use_streaming: If True, stream dataset sample-by-sample (recommended,
                low disk usage). If False, downloads the FULL dataset to local
                HF cache first (10-45GB) -- only use this if you have the space
                and streaming is somehow unavailable.
            target_sr: Target sampling rate (None to keep original)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_name = dataset_name
        self.use_streaming = use_streaming
        self.target_sr = target_sr

        load_dotenv()
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise RuntimeError(
                "HF_TOKEN not found in .env file. "
                "Copy .env.example to .env and add your own token."
            )

        self._dataset = None
        print(f"AudioLoader initialized. Cache directory: {self.cache_dir}")

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _get_cache_path(self, clip_id: str) -> Path:
        return self.cache_dir / f"{clip_id}.wav"

    def _load_from_cache(self, clip_id: str) -> Optional[Tuple[np.ndarray, int]]:
        cache_path = self._get_cache_path(clip_id)
        if cache_path.exists():
            try:
                audio, sr = sf.read(cache_path)
                return audio, sr
            except Exception as e:
                print(f"Warning: failed to load cached file {cache_path}: {e}")
                cache_path.unlink(missing_ok=True)
                return None
        return None

    def _save_to_cache(self, clip_id: str, audio: np.ndarray, sr: int):
        cache_path = self._get_cache_path(clip_id)
        try:
            sf.write(cache_path, audio, sr)
        except Exception as e:
            print(f"Warning: failed to save to cache {cache_path}: {e}")

    # ------------------------------------------------------------------ #
    # Dataset access
    # ------------------------------------------------------------------ #

    def _get_dataset(self):
        if self._dataset is None:
            print("Loading Vaani dataset from HuggingFace (streaming mode)...")
            self._dataset = load_dataset(
                self.dataset_name,
                token=self.token,
                streaming=self.use_streaming,
            )
        return self._dataset

    def _extract_index_from_clip_id(self, clip_id: str) -> int:
        """
        "train_000042" -> 42
        """
        try:
            return int(clip_id.replace("train_", ""))
        except ValueError:
            raise ValueError(f"Invalid clip_id format: {clip_id}. Expected 'train_XXXXXX'")

    def _fetch_from_hf(self, clip_id: str) -> Tuple[np.ndarray, int]:
        """
        Fetch audio from HuggingFace by clip_id.

        Uses .skip(target_index) to jump close to the target row instead of
        iterating one-by-one from the start -- much faster for high indices.
        Does NOT cast the audio column, so `datasets` uses its default
        decoder (soundfile-based) instead of torchcodec.
        """
        dataset = self._get_dataset()
        train_split = dataset["train"]
        target_index = self._extract_index_from_clip_id(clip_id)

        # Force decode=False so `datasets` gives us raw encoded bytes instead
        # of trying to decode audio itself (which would require torchcodec).
        train_split = train_split.cast_column("audio", Audio(decode=False))

        if self.use_streaming:
            try:
                skipped = train_split.skip(target_index)
                sample = next(iter(skipped))

                audio_array, sr = self._decode_raw_audio(sample["audio"])

                if self.target_sr is not None and sr != self.target_sr:
                    audio_array, sr = self._resample(audio_array, sr, self.target_sr)

                return audio_array, sr

            except StopIteration:
                raise ValueError(
                    f"Could not find clip_id {clip_id} (index {target_index}) "
                    "in dataset - index out of range"
                )
            except Exception as e:
                raise RuntimeError(f"Error fetching {clip_id} from HuggingFace: {e}")

        else:
            # Non-streaming: dataset is fully downloaded locally first (large!).
            # Only use this path deliberately, not as a silent fallback.
            sample = train_split[target_index]
            audio_array, sr = self._decode_raw_audio(sample["audio"])

            if self.target_sr is not None and sr != self.target_sr:
                audio_array, sr = self._resample(audio_array, sr, self.target_sr)

            return audio_array, sr

    def _decode_raw_audio(self, audio_field: dict) -> Tuple[np.ndarray, int]:
        """
        Decode raw encoded audio bytes (from Audio(decode=False)) using
        soundfile -- avoids any torchcodec/FFmpeg dependency entirely.
        """
        raw_bytes = audio_field["bytes"]
        audio_array, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        return audio_array, sr

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
        import torch

        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio_resampled = resampler(audio_tensor)

        return audio_resampled.squeeze().numpy(), target_sr

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_audio(self, clip_id: str, force_reload: bool = False) -> Tuple[np.ndarray, int]:
        """
        Load audio for a clip_id, using local cache when available.

        Example:
            >>> loader = AudioLoader()
            >>> audio, sr = loader.load_audio("train_000042")
        """
        if not force_reload:
            cached = self._load_from_cache(clip_id)
            if cached is not None:
                return cached

        print(f"Fetching {clip_id} from HuggingFace...")
        audio, sr = self._fetch_from_hf(clip_id)
        self._save_to_cache(clip_id, audio, sr)
        return audio, sr

    def batch_load(self, clip_ids: list, show_progress: bool = True) -> dict:
        results = {}
        iterator = tqdm(clip_ids, desc="Loading audio") if show_progress else clip_ids
        for clip_id in iterator:
            try:
                results[clip_id] = self.load_audio(clip_id)
            except Exception as e:
                print(f"Error loading {clip_id}: {e}")
                results[clip_id] = None
        return results

    def clear_cache(self, clip_id: Optional[str] = None):
        if clip_id is not None:
            self._get_cache_path(clip_id).unlink(missing_ok=True)
            print(f"Cleared cache for {clip_id}")
        else:
            for f in self.cache_dir.glob("*.wav"):
                f.unlink()
            print(f"Cleared all cached audio files from {self.cache_dir}")

    def get_cache_stats(self) -> dict:
        cache_files = list(self.cache_dir.glob("*.wav"))
        total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        return {
            "num_cached": len(cache_files),
            "total_size_mb": round(total_size_mb, 2),
            "cache_dir": str(self.cache_dir),
        }


if __name__ == "__main__":
    print("=== AudioLoader Test ===\n")
    loader = AudioLoader(cache_dir="data/cache/audio_clips", target_sr=16000)

    test_clip_ids = ["train_000001", "train_000002", "train_000005"]
    print("Loading test clips...")
    for clip_id in test_clip_ids:
        try:
            audio, sr = loader.load_audio(clip_id)
            print(f"{clip_id}: shape={audio.shape}, sr={sr}, duration={len(audio)/sr:.2f}s")
        except Exception as e:
            print(f"Error loading {clip_id}: {e}")

    print("\nCache statistics:")
    stats = loader.get_cache_stats()
    print(f"  Cached files: {stats['num_cached']}")
    print(f"  Total size: {stats['total_size_mb']} MB")
    print(f"  Cache directory: {stats['cache_dir']}")