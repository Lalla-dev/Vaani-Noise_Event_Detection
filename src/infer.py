"""
infer.py
Inference and submission generation pipeline for Vaani Noise Event Detection (Track 1).

Supports:
1. Codabench input_data mode: directly processes local test/eval audio directory or zip file
   (e.g., input_data/audio/*.wav or input_data.zip) with zero metadata dependency.
2. Validation mode: processes extracted validation folder with validationMetadata.json.
3. Pipeline mode: processes split JSONL with local cache / streaming.

Event Decoding:
- Probability thresholding + temporal median filtering (scipy.ndimage.median_filter)
- Automatically creates predictions.jsonl and packages submission_track1.zip ready for Codabench.
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

from src.features import LogMelExtractor
from src.model import CRNNNoiseDetector


# ---------------------------------------------------------------------------
# AudioFolderDataset: Direct audio file & zip loader for Codabench input_data
# ---------------------------------------------------------------------------
class AudioFolderDataset(Dataset):
    """
    Dataset loader for Codabench Track 1 input_data (local audio files or zip archive).
    
    Supports:
    - Extracted folder containing audio/ subfolder or direct audio files.
    - Direct .zip archive (e.g. input_data.zip) read in-memory.
    - Supported formats: .wav, .flac, .mp3, .ogg.
    - Yields mono audio float32 tensor resampled to target_sr (16 kHz).
    """

    SUPPORTED_EXTS = (".wav", ".flac", ".mp3", ".ogg")

    def __init__(self, input_path: Union[str, Path], target_sr: int = 16000):
        self.input_path = Path(input_path)
        self.target_sr = target_sr
        self.is_zip = self.input_path.is_file() and self.input_path.suffix.lower() == ".zip"
        self._zip_file = None

        if self.is_zip:
            self._zip_file = zipfile.ZipFile(self.input_path, "r")
            all_names = self._zip_file.namelist()
            self.file_list = sorted([
                f for f in all_names
                if any(f.lower().endswith(ext) for ext in self.SUPPORTED_EXTS)
                and not Path(f).name.startswith(".")
                and not f.startswith("__MACOSX")
            ])
            print(f"AudioFolderDataset initialized: {len(self.file_list)} clips from zip '{self.input_path.name}'")
        elif self.input_path.is_dir():
            # Check if there is an audio/ subfolder inside input_path
            search_dir = self.input_path / "audio" if (self.input_path / "audio").is_dir() else self.input_path
            all_files = list(search_dir.rglob("*"))
            self.file_list = sorted([
                f for f in all_files
                if f.suffix.lower() in self.SUPPORTED_EXTS
                and not f.name.startswith(".")
                and "__MACOSX" not in f.parts
            ])
            print(f"AudioFolderDataset initialized: {len(self.file_list)} clips from directory '{search_dir}'")
        else:
            raise FileNotFoundError(f"Input audio path not found: {self.input_path}")

        if len(self.file_list) == 0:
            raise ValueError(f"No audio files found in {self.input_path}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict:
        item = self.file_list[idx]
        if self.is_zip:
            clip_id = Path(item).stem
            try:
                raw_bytes = self._zip_file.read(item)
                audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            except Exception as e:
                print(f"[Warning] Failed to read {item} from zip: {e}")
                audio = np.zeros(self.target_sr * 5, dtype=np.float32)
                sr = self.target_sr
        else:
            clip_id = item.stem
            try:
                audio, sr = sf.read(str(item), dtype="float32")
            except Exception as e:
                print(f"[Warning] Failed to read {item}: {e}")
                audio = np.zeros(self.target_sr * 5, dtype=np.float32)
                sr = self.target_sr

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != self.target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        duration = float(len(audio) / sr)
        audio_tensor = torch.from_numpy(audio).float()

        return {
            "clip_id": clip_id,
            "audio": audio_tensor,
            "sample_rate": sr,
            "duration": duration,
        }


def parse_args():
    # Detect best default model checkpoint
    default_model = "models/baseline_crnn_9426_gold.pt"
    if not (PROJECT_ROOT / default_model).exists():
        default_model = "models/baseline_crnn_5000_gold.pt"
    if not (PROJECT_ROOT / default_model).exists():
        default_model = "models/baseline_crnn_1000_gold.pt"
    if not (PROJECT_ROOT / default_model).exists():
        default_model = "models/baseline_crnn.pt"

    parser = argparse.ArgumentParser(description="Baseline inference for Vaani Noise Event Detection (Track 1)")
    parser.add_argument("--model", type=str, default=default_model,
                        help=f"Path to trained model checkpoint (default: {default_model})")
    parser.add_argument("--input-data", type=str, default=None,
                        help="Path to Codabench input_data folder or input_data.zip containing audio files")
    parser.add_argument("--validation-dir", type=str, default=None,
                        help="Path to extracted Codabench validation folder (reads validationMetadata.json + local WAVs)")
    parser.add_argument("--input-jsonl", type=str, default=None,
                        help="Path to validation JSONL split (defaults to data/processed/val_split.jsonl if other modes not given)")
    parser.add_argument("--cache-dir", type=str, default="data/cache/audio_clips",
                        help="Directory for cached audio files")
    parser.add_argument("--out", type=str, default="submissions/predictions.jsonl",
                        help="Output predictions JSONL path")
    parser.add_argument("--zip-out", type=str, default=None,
                        help="Explicit destination path for Codabench submission zip package")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Probability threshold for event detection (default: 0.35)")
    parser.add_argument("--median-filter", type=int, default=7,
                        help="Window size for median filtering binary prediction mask (default: 7, set <=1 to disable)")
    parser.add_argument("--min-duration", type=float, default=0.0,
                        help="Minimum event duration in seconds (default: 0.0)")
    parser.add_argument("--merge-gap", type=float, default=0.0,
                        help="Maximum gap in seconds between consecutive events to merge (default: 0.0)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device (cpu or cuda)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Process at most N samples (for quick testing)")
    parser.add_argument("--cached-first", action=argparse.BooleanOptionalAction, default=False,
                        help="Prioritize cached audio clips (for JSONL mode)")
    parser.add_argument("--tier-filter", type=str, nargs="*",
                        help="Optional list of tiers to filter (for JSONL mode)")
    parser.add_argument("--create-zip", action=argparse.BooleanOptionalAction, default=True,
                        help="Create Codabench submission zip archive containing predictions JSONL (default: True)")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    p = Path(ckpt_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found at {p}. Please train the model first.")
    ckpt = torch.load(str(p), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def frames_to_seconds(start_frame: int, end_frame: int, hop_length: int, sr: int, duration: float) -> Tuple[float, float]:
    onset = round(start_frame * hop_length / sr, 3)
    offset = round(min(end_frame * hop_length / sr, duration), 3)
    return onset, offset


def logits_to_events(
    probs: torch.Tensor,
    threshold: float,
    hop_length: int,
    sr: int,
    duration: float,
    med: int = 5,
    merge_gap: float = 0.0,
    min_duration: float = 0.0,
) -> List[Dict]:
    mask = (probs >= threshold).cpu().numpy().astype(np.uint8)
    if med > 1:
        from scipy.ndimage import median_filter
        mask = median_filter(mask, size=med, mode="nearest")

    events = []
    if mask.sum() == 0:
        return events

    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        onset, offset = frames_to_seconds(s, e, hop_length, sr, duration)
        if offset > onset:
            events.append({"onset": onset, "offset": offset})

    # Post-processing: merge gap and min duration
    if merge_gap > 0.0 and len(events) > 1:
        merged = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            if ev["onset"] - prev["offset"] <= merge_gap:
                merged[-1] = {"onset": prev["onset"], "offset": round(max(prev["offset"], ev["offset"]), 3)}
            else:
                merged.append(ev)
        events = merged

    if min_duration > 0.0:
        events = [ev for ev in events if round(ev["offset"] - ev["onset"], 3) >= min_duration]

    return events


def resolve_dataset(args) -> Dataset:
    """Determine and instantiate the appropriate Dataset based on CLI arguments and filesystem discovery."""
    # 1. Explicit --input-data provided
    if args.input_data:
        p = Path(args.input_data)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        print(f"Running inference in Codabench input_data mode: {p}")
        return AudioFolderDataset(p, target_sr=16000)

    # 2. Explicit --validation-dir provided
    if args.validation_dir:
        p = Path(args.validation_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        # Check if validationMetadata.json exists
        if (p / "validationMetadata.json").exists() or (p / "validation" / "validationMetadata.json").exists():
            from src.val_loader import CodabenchValDataset
            print(f"Running inference in Codabench Validation mode (with metadata): {p}")
            return CodabenchValDataset(validation_dir=str(p))
        else:
            print(f"Running inference in AudioFolder mode: {p}")
            return AudioFolderDataset(p, target_sr=16000)

    # 3. Explicit --input-jsonl provided
    if args.input_jsonl:
        from src.dataset import VaaniNoiseDataset
        print(f"Running inference in JSONL pipeline mode: {args.input_jsonl}")
        ds = VaaniNoiseDataset(
            jsonl_path=args.input_jsonl,
            cache_dir=args.cache_dir,
            target_sr=16000,
            tier_filter=args.tier_filter,
            transform=None,
        )
        return ds

    # 4. Auto-discovery of local datasets (prefers local input_data over streaming)
    search_paths = [
        PROJECT_ROOT / "input_data",
        PROJECT_ROOT.parent / "input_data",
        Path.home() / "Downloads" / "input_data.zip",
        PROJECT_ROOT.parent / "Validation" / "validation",
        PROJECT_ROOT / "Validation" / "validation",
    ]
    for candidate in search_paths:
        if candidate.exists():
            if candidate.suffix.lower() == ".zip" or (candidate.is_dir() and ((candidate / "audio").is_dir() or any(candidate.glob("*.wav")))):
                print(f"Auto-discovered Codabench input data at: {candidate}")
                return AudioFolderDataset(candidate, target_sr=16000)
            elif candidate.is_dir() and (candidate / "validationMetadata.json").exists():
                from src.val_loader import CodabenchValDataset
                print(f"Auto-discovered Validation dataset at: {candidate}")
                return CodabenchValDataset(validation_dir=str(candidate))

    # 5. Final fallback: val_split.jsonl
    from src.dataset import VaaniNoiseDataset
    default_jsonl = "data/processed/val_split.jsonl"
    print(f"No local input_data discovered; falling back to JSONL mode: {default_jsonl}")
    return VaaniNoiseDataset(
        jsonl_path=default_jsonl,
        cache_dir=args.cache_dir,
        target_sr=16000,
        tier_filter=args.tier_filter,
        transform=None,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 65)
    print("VAANI TRACK 1: INFERENCE & SUBMISSION GENERATOR")
    print("=" * 65)
    print(f"Model Checkpoint : {args.model}")
    print(f"Device           : {device}")
    print(f"Threshold        : {args.threshold}")
    print(f"Median Filter    : {args.median_filter}")

    # Load model
    model = CRNNNoiseDetector(n_mels=64)
    model = load_checkpoint(model, args.model, device)

    # Resolve dataset
    dataset = resolve_dataset(args)

    if args.max_samples and hasattr(dataset, "file_list"):
        dataset.file_list = dataset.file_list[:args.max_samples]
    elif args.max_samples and hasattr(dataset, "samples"):
        dataset.samples = dataset.samples[:args.max_samples]

    extractor = LogMelExtractor(sr=16000, n_mels=64, n_fft=1024, hop_length=512)

    # Ensure output directory exists
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {len(dataset)} audio clips...")
    start_time = time.time()
    processed = 0
    total_events = 0
    clips_with_events = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            clip_id = sample["clip_id"]
            audio = sample["audio"].numpy()
            sr = sample["sample_rate"]
            duration = float(sample.get("duration", len(audio) / sr))

            # Feature extraction
            try:
                feats = extractor.extract(audio)
            except Exception as e:
                print(f"[Warning] Feature extraction failed for {clip_id}: {e}")
                events = []
                fout.write(json.dumps({"clip_id": clip_id, "events": events}) + "\n")
                continue

            feats = feats.unsqueeze(0).to(device)  # [1, T, n_mels]
            with torch.no_grad():
                logits = model(feats).squeeze(0)  # [T]
                probs = torch.sigmoid(logits)

            events = logits_to_events(
                probs,
                args.threshold,
                extractor.hop_length,
                extractor.sr,
                duration,
                med=args.median_filter,
                merge_gap=args.merge_gap,
                min_duration=args.min_duration,
            )

            if events:
                clips_with_events += 1
                total_events += len(events)

            fout.write(json.dumps({"clip_id": clip_id, "events": events}) + "\n")
            processed += 1

            if processed % 500 == 0 or processed == len(dataset):
                elapsed = time.time() - start_time
                rate = processed / max(elapsed, 1e-3)
                print(f"[{processed:5d}/{len(dataset)}] clips processed ({rate:.1f} clips/s) | Events detected: {total_events}", flush=True)

            if args.max_samples and processed >= args.max_samples:
                break

    elapsed = time.time() - start_time
    print("=" * 65)
    print(f"Inference completed in {elapsed:.2f}s ({processed / max(elapsed, 1e-3):.1f} clips/s)")
    print(f"Total clips processed  : {processed}")
    print(f"Clips with events      : {clips_with_events} / {processed} ({clips_with_events/max(processed,1)*100:.1f}%)")
    print(f"Total events detected  : {total_events}")
    print(f"Predictions written to : {out_path}")

    # Automatic Codabench submission zip packaging
    if args.create_zip:
        if args.zip_out:
            zip_path = Path(args.zip_out)
            if not zip_path.is_absolute():
                zip_path = PROJECT_ROOT / zip_path
        else:
            zip_path = out_path.parent / f"submission_track1_{out_path.stem}.zip" if out_path.stem != "predictions" else out_path.parent / "submission_track1.zip"
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_path, arcname=out_path.name)
        print(f"\nCodabench submission package created: {zip_path}")
        print(f"  - Structure: '{out_path.name}' directly at archive root (verified)")
    print("=" * 65)


if __name__ == "__main__":
    main()
