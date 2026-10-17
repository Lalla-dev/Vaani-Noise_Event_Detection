"""CLI runner for Track 2 SraVaani transcription.

Transcribes enhanced WAV files into transcripts.jsonl.
Resumable, robust, uses FP16 for maximum GPU throughput.
"""

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from track2.transcribe import SraVaaniTranscriber, run_batch_transcription


def main():
    parser = argparse.ArgumentParser(description="Track 2 SraVaani Transcription Runner")
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=str(repo_root / "data" / "enhanced_track2_phase2"),
        help="Path to enhanced audio folder",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(repo_root / "data" / "transcripts.jsonl"),
        help="Path to output transcripts.jsonl",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for ASR inference",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Logging interval in clips",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("TRACK 2 PHASE 2 ASR TRANSCRIPTION RUNNER")
    print("=" * 65)
    print(f"Audio Directory: {args.audio_dir}")
    print(f"Output File:     {args.output_file}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Mandated Model:  ARTPARK-IISc/SraVaani-1.0 (FP16)")
    print("=" * 65)

    transcriber = SraVaaniTranscriber()
    summary = run_batch_transcription(
        audio_dir=args.audio_dir,
        output_jsonl=args.output_file,
        transcriber=transcriber,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )

    print("\nTranscription complete. Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
