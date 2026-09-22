"""
val_loader.py
Validation Dataset adapter for Codabench Track 1 (Noise Event Detection).

Loads audio and metadata directly from the extracted Codabench validation folder:
- validationMetadata.json (JSON array of 2,501 entries)
- naturalNoisyAudio/ (for natural clips, syntheticData == False)
- syntheticNoiseAudio/ (for synthetic clips, syntheticData == True)
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class CodabenchValDataset(Dataset):
    """
    Dataset loader for Codabench Noise Event Detection validation dataset.
    
    Reads validationMetadata.json and loads local WAV audio directly without
    relying on Hugging Face streaming or the training audio cache.
    """

    def __init__(
        self,
        validation_dir: Union[str, Path],
        target_sr: int = 16000,
        synthetic_only: bool = False,
        natural_only: bool = False,
    ):
        """
        Args:
            validation_dir: Path to the validation folder (contains validationMetadata.json,
                            naturalNoisyAudio/, syntheticNoiseAudio/).
            target_sr: Expected audio sample rate (validation WAVs are already 16 kHz).
            synthetic_only: If True, only load synthetic samples.
            natural_only: If True, only load natural samples.
        """
        self.validation_dir = Path(validation_dir)
        
        # Handle nested validation/validation directory if user provided outer directory
        if (self.validation_dir / "validation" / "validationMetadata.json").exists():
            self.validation_dir = self.validation_dir / "validation"

        self.metadata_path = self.validation_dir / "validationMetadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"validationMetadata.json not found in {self.validation_dir}"
            )

        self.natural_dir = self.validation_dir / "naturalNoisyAudio"
        self.synthetic_dir = self.validation_dir / "syntheticNoiseAudio"
        self.clean_ref_dir = self.validation_dir / "syntheticCleanRefAudio"
        self.target_sr = target_sr

        self.samples = self._load_metadata(synthetic_only, natural_only)
        print(f"CodabenchValDataset initialized: {len(self.samples)} samples from {self.metadata_path}")

    def _load_metadata(self, synthetic_only: bool, natural_only: bool) -> List[Dict]:
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for entry in raw_data:
            is_synthetic = bool(entry.get("syntheticData", False))
            if synthetic_only and not is_synthetic:
                continue
            if natural_only and is_synthetic:
                continue

            fname = entry["segmentFileName"]
            # Extract stem matching competition clip_id specification
            clip_id = Path(fname).stem

            # Resolve local audio path
            if is_synthetic:
                audio_path = self.synthetic_dir / fname
            else:
                audio_path = self.natural_dir / fname

            # Parse ground truth events into standardized onset/offset floats
            events = []
            raw_events = entry.get("NoiseSubCategoryTimeStamp", [])
            if isinstance(raw_events, list):
                for ev in raw_events:
                    if isinstance(ev, dict) and "start" in ev and "end" in ev:
                        try:
                            onset = float(ev["start"])
                            offset = float(ev["end"])
                            if offset > onset:
                                events.append({
                                    "onset": onset,
                                    "offset": offset,
                                    "category": ev.get("category", "unknown"),
                                    "tag": ev.get("tag", ""),
                                })
                        except (ValueError, TypeError):
                            continue

            samples.append({
                "clip_id": clip_id,
                "segment_file_name": fname,
                "audio_path": audio_path,
                "duration": float(entry.get("duration", 0.0)),
                "synthetic": is_synthetic,
                "language": entry.get("language", "unknown"),
                "state": entry.get("state", ""),
                "district": entry.get("district", ""),
                "transcript": entry.get("transcript", ""),
                "categories": entry.get("NoiseCategory", []),
                "events": events,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        audio_path = sample["audio_path"]

        try:
            audio, sr = sf.read(str(audio_path), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
        except Exception as e:
            print(f"[Warning] Failed to load audio for {sample['clip_id']} ({audio_path}): {e}")
            dur = sample["duration"] if sample["duration"] > 0 else 5.0
            audio = np.zeros(int(self.target_sr * dur), dtype=np.float32)
            sr = self.target_sr

        if self.target_sr is not None and sr != self.target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        audio_tensor = torch.from_numpy(audio).float()
        duration = sample["duration"] if sample["duration"] > 0 else float(len(audio) / sr)

        return {
            "audio": audio_tensor,
            "sample_rate": sr,
            "duration": duration,
            "clip_id": sample["clip_id"],
            "segment_file_name": sample["segment_file_name"],
            "synthetic": sample["synthetic"],
            "language": sample["language"],
            "events": sample["events"],
        }
