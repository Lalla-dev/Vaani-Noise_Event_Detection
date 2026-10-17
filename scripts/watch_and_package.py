"""Watch transcription progress and immediately package & validate Track 2 ZIP upon completion."""

import json
import time
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from track2.package_submission import validate_and_package_submission

INPUT_DIR = Path(r"C:\Users\Shubham\Desktop\vaani project\input_data_track-2_phase-2\audio")
ENHANCED_DIR = repo_root / "data" / "enhanced_track2_phase2"
TRANSCRIPTS_FILE = repo_root / "data" / "transcripts.jsonl"
OUTPUT_ZIP = repo_root / "submissions" / "submission_track2_phase2.zip"
EXPECTED_COUNT = 3178

print(f"Watchdog active. Waiting for transcripts.jsonl to reach {EXPECTED_COUNT} records...", flush=True)

while True:
    count = 0
    if TRANSCRIPTS_FILE.exists():
        with open(TRANSCRIPTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    
    if count >= EXPECTED_COUNT:
        print(f"\nTarget reached: {count}/{EXPECTED_COUNT} transcripts! Triggering packaging immediately...", flush=True)
        break
    
    time.sleep(3)

# Execute packaging and validation
result = validate_and_package_submission(
    input_audio_dir=INPUT_DIR,
    enhanced_audio_dir=ENHANCED_DIR,
    transcripts_file=TRANSCRIPTS_FILE,
    output_zip=OUTPUT_ZIP,
)

print("\nWATCHDOG PACKAGING RESULT:")
print(json.dumps(result, indent=2))
