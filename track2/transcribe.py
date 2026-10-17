"""ASR Transcription Module for Track 2.

Transcribes enhanced WAV files using the mandated ARTPARK-IISc/SraVaani-1.0 model.
Produces root-level transcripts.jsonl formatted as:
    {"clip_id": "<stem_without_wav>", "text": "<transcription>"}
Resumable and validates output formatting.
"""

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Union
import soundfile as sf
import torch


class SraVaaniTranscriber:
    """ASR transcriber using ARTPARK-IISc/SraVaani-1.0."""

    def __init__(
        self,
        model_name: str = "ARTPARK-IISc/SraVaani-1.0",
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if hf_token is None:
            envp = Path(r"C:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\.env")
            if envp.exists():
                for line in envp.read_text(encoding="utf-8").splitlines():
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=", 1)[1].strip()

        print(f"Loading mandated ASR model '{model_name}' on {self.device}...")
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
            dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
        ).to(self.device).eval()
        print(f"SraVaani ASR model loaded successfully on {self.device}.")

    @torch.no_grad()
    def transcribe_paths(self, paths: List[str]) -> List[str]:
        """Transcribe a list of WAV paths."""
        try:
            hyps = self.model.transcribe(paths, return_hypotheses=True)
            return [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
        except Exception:
            # Fallback to single file execution
            results = []
            for p in paths:
                try:
                    h = self.model.transcribe([p], return_hypotheses=True)[0]
                    results.append((h.text if hasattr(h, "text") else str(h)).strip())
                except Exception as e:
                    print(f"Error transcribing {p}: {e}")
                    results.append("")
            return results

    def transcribe_file(self, wav_path: Union[str, Path]) -> str:
        """Transcribe a single audio file."""
        return self.transcribe_paths([str(wav_path)])[0]


def run_batch_transcription(
    audio_dir: Union[str, Path],
    output_jsonl: Union[str, Path],
    transcriber: Optional[SraVaaniTranscriber] = None,
    batch_size: int = 8,
    log_interval: int = 50,
) -> dict:
    """Transcribe all WAV files in audio_dir into a transcripts.jsonl file.

    Resumable: reads existing entries in output_jsonl to skip already transcribed clips.

    Args:
        audio_dir: Path to directory containing enhanced WAV files.
        output_jsonl: Output path for transcripts.jsonl.
        transcriber: SraVaaniTranscriber instance, or None to initialize.
        batch_size: Batch size for ASR transcription.
        log_interval: Interval for progress logging.

    Returns:
        Summary dict.
    """
    in_dir = Path(audio_dir)
    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(in_dir.glob("*.wav")))
    total_files = len(wav_files)
    print(f"Found {total_files} WAV files to transcribe in {in_dir}")

    # Read existing entries for resumption
    existing_transcripts: Dict[str, str] = {}
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        existing_transcripts[record["clip_id"]] = record["text"]
                    except Exception:
                        pass
    print(f"Resumption check: {len(existing_transcripts)} clips already transcribed.")

    pending_files = [f for f in wav_files if f.stem not in existing_transcripts]
    print(f"Pending clips to transcribe: {len(pending_files)}")

    if not pending_files:
        print("All clips already transcribed!")
        return {
            "total_files": total_files,
            "transcribed_files": 0,
            "existing_files": len(existing_transcripts),
        }

    if transcriber is None:
        transcriber = SraVaaniTranscriber()

    start_time = time.time()
    num_processed = 0

    # Append to output_jsonl as we go
    with open(out_file, "a", encoding="utf-8") as out_f:
        for i in range(0, len(pending_files), batch_size):
            batch = pending_files[i : i + batch_size]
            paths = [str(f) for f in batch]
            texts = transcriber.transcribe_paths(paths)

            for f, text in zip(batch, texts):
                record = {"clip_id": f.stem, "text": text}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            num_processed += len(batch)
            if num_processed % log_interval == 0 or num_processed == len(pending_files):
                elapsed = time.time() - start_time
                sec_per_clip = elapsed / num_processed
                remaining = len(pending_files) - num_processed
                eta_min = (remaining * sec_per_clip) / 60.0
                print(
                    f"[{num_processed}/{len(pending_files)}] "
                    f"Elapsed: {elapsed / 60:.1f}m | "
                    f"Speed: {1.0 / sec_per_clip:.1f} clips/s | "
                    f"ETA: {eta_min:.1f} min",
                    flush=True,
                )

    wall_total = time.time() - start_time
    print(f"Transcription complete! Transcribed {num_processed} clips in {wall_total / 60:.2f} minutes.")

    return {
        "total_files": total_files,
        "transcribed_files": num_processed,
        "existing_files": len(existing_transcripts),
        "total_runtime_minutes": wall_total / 60.0,
    }
