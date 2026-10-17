"""Packaging and Validation for Track 2 Submission.

Creates submissions/submission_track2_phase2.zip with:
1. Exactly 3,178 enhanced 16 kHz mono PCM_16 WAV files at zip root.
2. Root-level transcripts.jsonl with exactly 3,178 records {"clip_id": "...", "text": "..."}.
3. 100% integrity validation (no nested folders, matching filenames, valid WAV headers).
"""

from pathlib import Path
import json
import zipfile
import soundfile as sf
from typing import Dict, List, Set


def validate_and_package_submission(
    input_audio_dir: Path,
    enhanced_audio_dir: Path,
    transcripts_file: Path,
    output_zip: Path,
) -> Dict:
    """Validate all files and create submission ZIP.

    Args:
        input_audio_dir: Original Phase 2 test audio directory.
        enhanced_audio_dir: Directory containing enhanced WAV files.
        transcripts_file: Path to transcripts.jsonl.
        output_zip: Path to output ZIP file.

    Returns:
        dict of validation results.
    """
    print("=" * 65)
    print("VALIDATING TRACK 2 SUBMISSION ARTIFACTS")
    print("=" * 65)

    input_wavs = sorted(list(input_audio_dir.glob("*.wav")))
    expected_count = len(input_wavs)
    expected_stems = {f.stem for f in input_wavs}
    print(f"Expected clips count from input: {expected_count}")

    # 1. Validate enhanced WAV files
    enhanced_wavs = sorted(list(enhanced_audio_dir.glob("*.wav")))
    print(f"Found enhanced WAV files: {len(enhanced_wavs)}")

    if len(enhanced_wavs) != expected_count:
        raise ValueError(
            f"Enhanced WAV count mismatch: found {len(enhanced_wavs)}, expected {expected_count}"
        )

    enhanced_stems = {f.stem for f in enhanced_wavs}
    missing_stems = expected_stems - enhanced_stems
    if missing_stems:
        raise ValueError(f"Missing enhanced WAVs for stems: {list(missing_stems)[:10]}")

    print("Verifying audio properties for all enhanced WAV files...")
    for idx, enh_path in enumerate(enhanced_wavs, 1):
        info = sf.info(str(enh_path))
        if info.samplerate != 16000:
            raise ValueError(f"File {enh_path.name} sample rate is {info.samplerate}, expected 16000")
        if info.channels != 1:
            raise ValueError(f"File {enh_path.name} channel count is {info.channels}, expected 1")
        if info.subtype != "PCM_16":
            raise ValueError(f"File {enh_path.name} subtype is {info.subtype}, expected PCM_16")

    print(f"All {len(enhanced_wavs)} WAV files verified (16 kHz, mono, PCM_16).")

    # 2. Validate transcripts.jsonl
    if not transcripts_file.exists():
        raise FileNotFoundError(f"transcripts.jsonl not found at {transcripts_file}")

    transcript_records = []
    seen_stems: Set[str] = set()

    with open(transcripts_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "clip_id" not in record or "text" not in record:
                raise ValueError(f"Invalid record at line {line_num}: {record}")

            cid = record["clip_id"]
            if cid in seen_stems:
                raise ValueError(f"Duplicate clip_id in transcripts: {cid}")
            seen_stems.add(cid)
            transcript_records.append(record)

    print(f"Found {len(transcript_records)} valid records in transcripts.jsonl.")
    if len(transcript_records) != expected_count:
        raise ValueError(
            f"Transcript records count ({len(transcript_records)}) != expected ({expected_count})"
        )

    missing_transcripts = expected_stems - seen_stems
    if missing_transcripts:
        raise ValueError(f"Missing transcripts for stems: {list(missing_transcripts)[:10]}")

    # 3. Create Submission ZIP
    print(f"\nBuilding submission ZIP at: {output_zip}")
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Add transcripts.jsonl at root
        print("Adding transcripts.jsonl to ZIP root...")
        zf.write(transcripts_file, arcname="transcripts.jsonl")

        # Add all WAV files at root
        print(f"Adding {len(enhanced_wavs)} enhanced WAV files to ZIP root...")
        for idx, enh_path in enumerate(enhanced_wavs, 1):
            zf.write(enh_path, arcname=enh_path.name)
            if idx % 500 == 0 or idx == len(enhanced_wavs):
                print(f"  Zipped [{idx}/{len(enhanced_wavs)}] WAVs...")

    # 4. Verify ZIP internal structure
    print("\nVerifying ZIP internal structure...")
    with zipfile.ZipFile(output_zip, "r") as zf:
        namelist = zf.namelist()
        print(f"Total entries in ZIP: {len(namelist)}")
        if len(namelist) != expected_count + 1:
            raise ValueError(
                f"ZIP entries count is {len(namelist)}, expected {expected_count + 1}"
            )
        if "transcripts.jsonl" not in namelist:
            raise ValueError("transcripts.jsonl is missing from ZIP root!")

        # Verify no directory prefixes
        for name in namelist:
            if "/" in name or "\\" in name:
                raise ValueError(f"Nested path detected in ZIP: {name}")

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print("=" * 65)
    print("SUBMISSION ZIP CREATION & VALIDATION SUCCESSFUL!")
    print(f"File: {output_zip}")
    print(f"Size: {zip_size_mb:.2f} MB")
    print(f"Entries: {len(namelist)} (1 transcripts.jsonl + {expected_count} .wav files)")
    print("=" * 65)

    return {
        "status": "VALID",
        "zip_path": str(output_zip),
        "zip_size_mb": zip_size_mb,
        "wav_count": len(enhanced_wavs),
        "transcript_count": len(transcript_records),
    }
