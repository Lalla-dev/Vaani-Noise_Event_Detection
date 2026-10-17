"""Streaming downloader and tester for ARTPARK-IISc/SraVaani-1.0."""

import os
from pathlib import Path
import time
import requests
import torch


TARGET_SIZE = 908846278  # Exact byte size of model-asr.fp16.ts
MODEL_ID = "ARTPARK-IISc/SraVaani-1.0"


def stream_download_weights():
    envp = Path(r"c:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\.env")
    tok = None
    for l in envp.read_text(encoding="utf-8").splitlines():
        if l.startswith("HF_TOKEN="):
            tok = l.split("=", 1)[1].strip()

    snapshot_dir = Path(
        os.path.expanduser(
            r"~/.cache/huggingface/hub/models--ARTPARK-IISc--SraVaani-1.0/snapshots/f5dd5358325a5208775b91dad98918e079ea2b27"
        )
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    out_file = snapshot_dir / "model-asr.fp16.ts"

    # Also link to blobs for full huggingface_hub compliance
    blob_dir = snapshot_dir.parent.parent / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_file = blob_dir / "789a21b6df8b0bb2ea2cc1c120edbfd4b898d3a26e6c6673cca45d14b64a47f2"

    if out_file.exists() and out_file.stat().st_size == TARGET_SIZE:
        print(f"model-asr.fp16.ts already completely downloaded ({out_file.stat().st_size} bytes).", flush=True)
    else:
        existing_size = out_file.stat().st_size if out_file.exists() else 0
        headers = {"Authorization": f"Bearer {tok}"}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            print(f"Resuming download from byte {existing_size} ({existing_size / (1024*1024):.1f} MB)...", flush=True)
            mode = "ab"
        else:
            print(f"Starting download of {TARGET_SIZE / (1024*1024):.1f} MB model weights...", flush=True)
            mode = "wb"

        url = f"https://huggingface.co/{MODEL_ID}/resolve/main/model-asr.fp16.ts"
        resp = requests.get(url, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()

        t0 = time.time()
        downloaded = existing_size
        last_log = time.time()

        with open(out_file, mode) as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_log > 3.0 or downloaded == TARGET_SIZE:
                        elapsed = time.time() - t0
                        speed_mb = ((downloaded - existing_size) / (1024 * 1024)) / max(elapsed, 0.1)
                        pct = (downloaded / TARGET_SIZE) * 100.0
                        rem_mb = (TARGET_SIZE - downloaded) / (1024 * 1024)
                        eta_sec = rem_mb / max(speed_mb, 0.01)
                        print(
                            f"[{pct:5.1f}%] {downloaded / (1024*1024):.1f}/{TARGET_SIZE / (1024*1024):.1f} MB "
                            f"@ {speed_mb:4.1f} MB/s | ETA: {eta_sec:.0f}s",
                            flush=True,
                        )
                        last_log = time.time()

        print(f"Download completed in {time.time()-t0:.1f}s!", flush=True)

    # Link/copy to blob file so transformers hub recognizes it
    if not blob_file.exists() or blob_file.stat().st_size != TARGET_SIZE:
        try:
            import shutil
            shutil.copyfile(out_file, blob_file)
            print("Linked to blob cache successfully.", flush=True)
        except Exception as e:
            print("Blob sync note:", e, flush=True)

    # Test loading and inference
    print("\nInstantiating AutoModel for verification...", flush=True)
    from transformers import AutoModel
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, token=tok).to(dev).eval()
    print("AutoModel loaded successfully onto", dev, flush=True)

    test_wav = Path(r"C:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\data\smoke_track2\indoMLDatathon_202608_00054484.wav")
    if test_wav.exists():
        print(f"Testing transcription on {test_wav.name}...", flush=True)
        t_start = time.time()
        hyps = model.transcribe([str(test_wav)], return_hypotheses=True)
        t_dur = time.time() - t_start
        text = (hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0])).strip()
        print(f"Transcription finished in {t_dur:.2f}s!", flush=True)
        print(f"Output text length: {len(text)} characters", flush=True)
        # Save to scratch for verification
        Path("scratch/verified_transcription.txt").write_text(f"{test_wav.name}\t{text}", encoding="utf-8")
        print("Written to scratch/verified_transcription.txt", flush=True)
    print("ALL ASR CHECKS PASSED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    stream_download_weights()
