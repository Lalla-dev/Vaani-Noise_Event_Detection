"""Final Track 2 Phase 2 Pipeline Runner.

1. Audits completed enhanced clips in data/enhanced_track2_phase2.
2. For unenhanced clips, fills with corresponding original WAV.
3. Transcribes all 3,178 clips using ARTPARK-IISc/SraVaani-1.0 on GPU with batching.
4. Validates exact 3,178 clip coverage in transcripts.jsonl.
5. Builds and validates submissions/submission_track2_phase2.zip.
"""

from pathlib import Path
import json
import shutil
import time
import zipfile
import soundfile as sf
import torch
from transformers import AutoModel


INPUT_DIR = Path(r"C:\Users\Shubham\Desktop\vaani project\input_data_track-2_phase-2\audio")
ENHANCED_DIR = Path(r"C:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\data\enhanced_track2_phase2")
TRANSCRIPTS_FILE = Path(r"C:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\data\transcripts.jsonl")
OUTPUT_ZIP = Path(r"C:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\submissions\submission_track2_phase2.zip")


def prepare_audio_pool():
    print("=" * 65)
    print("STEP 1: AUDITING AND PREPARING AUDIO POOL")
    print("=" * 65)
    all_input_wavs = sorted(list(INPUT_DIR.glob("*.wav")))
    total_expected = len(all_input_wavs)
    print(f"Total expected Phase 2 test clips: {total_expected}")

    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    existing_enhanced = {f.name: f for f in ENHANCED_DIR.glob("*.wav")}

    # Validate existing enhanced files
    valid_enhanced_names = set()
    for name, path in existing_enhanced.items():
        try:
            info = sf.info(str(path))
            if info.frames > 0 and info.samplerate == 16000 and info.subtype == "PCM_16":
                valid_enhanced_names.add(name)
            else:
                path.unlink()
        except Exception:
            path.unlink()

    denoised_count = len(valid_enhanced_names)
    fallback_count = 0

    print(f"Valid SepFormer enhanced clips on disk: {denoised_count} / {total_expected}")

    # Copy fallback for any missing clip
    for in_wav in all_input_wavs:
        if in_wav.name not in valid_enhanced_names:
            dest = ENHANCED_DIR / in_wav.name
            shutil.copyfile(in_wav, dest)
            fallback_count += 1

    total_ready = len(list(ENHANCED_DIR.glob("*.wav")))
    print(f"Total audio clips in pool: {total_ready} ({denoised_count} SepFormer denoised, {fallback_count} raw fallback)")
    assert total_ready == total_expected, f"Count mismatch: {total_ready} != {total_expected}"
    return total_expected, denoised_count, fallback_count


def run_transcription(batch_size=16):
    print("\n" + "=" * 65)
    print("STEP 2: RUNNING SRAVAANI-1.0 ASR TRANSCRIPTION")
    print("=" * 65)
    all_wavs = sorted(list(ENHANCED_DIR.glob("*.wav")))
    total_clips = len(all_wavs)

    # Load existing transcripts for resumption
    existing_records = {}
    if TRANSCRIPTS_FILE.exists():
        with open(TRANSCRIPTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("clip_id") and rec.get("text"):
                            existing_records[rec["clip_id"]] = rec["text"]
                    except Exception:
                        pass

    print(f"Resumption check: {len(existing_records)} clips already transcribed.")
    pending_wavs = [w for w in all_wavs if w.stem not in existing_records]
    print(f"Pending clips to transcribe: {len(pending_wavs)}")

    if not pending_wavs:
        print("All clips already transcribed!")
        return

    # Load SraVaani on GPU
    envp = Path(r"C:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\.env")
    tok = None
    if envp.exists():
        for l in envp.read_text(encoding="utf-8").splitlines():
            if l.startswith("HF_TOKEN="):
                tok = l.split("=", 1)[1].strip()

    print("Loading ARTPARK-IISc/SraVaani-1.0 on cuda:0 (FP16)...", flush=True)
    model = AutoModel.from_pretrained(
        "ARTPARK-IISc/SraVaani-1.0",
        trust_remote_code=True,
        token=tok,
        dtype=torch.float16,
    ).to("cuda:0").eval()
    print("Model loaded successfully onto GPU.", flush=True)

    t_start = time.time()
    num_done = 0

    with open(TRANSCRIPTS_FILE, "a", encoding="utf-8") as out_f:
        for i in range(0, len(pending_wavs), batch_size):
            batch = pending_wavs[i : i + batch_size]
            paths = [str(w) for w in batch]
            hyps = model.transcribe(paths, return_hypotheses=True)

            for w, h in zip(batch, hyps):
                text = (h.text if hasattr(h, "text") else str(h)).strip()
                record = {"clip_id": w.stem, "text": text}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

            num_done += len(batch)
            elapsed = time.time() - t_start
            rate = num_done / max(elapsed, 0.1)
            rem_sec = (len(pending_wavs) - num_done) / max(rate, 0.01)
            print(
                f"[{num_done}/{len(pending_wavs)}] "
                f"Transcribed @ {rate:.2f} clips/s | "
                f"Elapsed: {elapsed/60:.1f}m | "
                f"ETA: {rem_sec/60:.1f}m",
                flush=True,
            )

    print(f"Transcription complete in {(time.time()-t_start)/60:.2f} minutes.")


def validate_and_package():
    print("\n" + "=" * 65)
    print("STEP 3: VALIDATING AND PACKAGING SUBMISSION ZIP")
    print("=" * 65)
    from track2.package_submission import validate_and_package_submission

    res = validate_and_package_submission(
        input_audio_dir=INPUT_DIR,
        enhanced_audio_dir=ENHANCED_DIR,
        transcripts_file=TRANSCRIPTS_FILE,
        output_zip=OUTPUT_ZIP,
    )
    print("\nFINAL RESULT:", res)
    return res


if __name__ == "__main__":
    prepare_audio_pool()
    run_transcription(batch_size=16)
    validate_and_package()
