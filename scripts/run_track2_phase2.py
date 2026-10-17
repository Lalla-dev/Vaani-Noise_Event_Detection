"""CLI execution script for Track 2 Phase 2 Audio Enhancement.

Enhances all Phase 2 test audio clips using SepFormer, saving to data/enhanced_track2_phase2.
Resumable, safe, and reports progress and ETA.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from track2.enhance import SepFormerEnhancer, run_batch_enhancement


def main():
    parser = argparse.ArgumentParser(description="Track 2 Phase 2 Speech Enhancement")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"C:\Users\Shubham\Desktop\vaani project\input_data_track-2_phase-2\audio",
        help="Path to Phase 2 test audio folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_root / "data" / "enhanced_track2_phase2"),
        help="Path to save enhanced audio files",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Interval of clips for logging progress and ETA",
    )
    parser.add_argument(
        "--dry-blend",
        type=float,
        default=0.0,
        help="Optional dry blend fraction of noisy input (default 0.0)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("TRACK 2 PHASE 2 SPEECH ENHANCEMENT PIPELINE")
    print("=" * 65)
    print(f"Input Directory:  {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Log Interval:     every {args.log_interval} clips")
    print(f"Backend:          SpeechBrain SepFormer (sepformer-dns4-16k-enhancement)")
    print("=" * 65)

    enhancer = SepFormerEnhancer()
    summary = run_batch_enhancement(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        enhancer=enhancer,
        log_interval=args.log_interval,
    )

    print("\nEnhancement complete. Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
