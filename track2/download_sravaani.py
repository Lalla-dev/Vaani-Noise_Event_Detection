"""Download and verify ARTPARK-IISc/SraVaani-1.0 model weights."""

from pathlib import Path
import time
from huggingface_hub import hf_hub_download
from transformers import AutoModel


def download_and_verify(token: str):
    print("Initiating download of ARTPARK-IISc/SraVaani-1.0 weights...", flush=True)
    t0 = time.time()
    path = hf_hub_download(
        repo_id="ARTPARK-IISc/SraVaani-1.0",
        filename="model-asr.fp16.ts",
        token=token,
        resume_download=True,
    )
    t1 = time.time()
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"model-asr.fp16.ts downloaded successfully: {size_mb:.2f} MB in {t1-t0:.1f}s", flush=True)

    print("Verifying model instantiation with AutoModel...", flush=True)
    model = AutoModel.from_pretrained(
        "ARTPARK-IISc/SraVaani-1.0",
        trust_remote_code=True,
        token=token,
    )
    print("AutoModel loaded successfully!", flush=True)
    return model


if __name__ == "__main__":
    envp = Path(r"c:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\.env")
    tok = None
    for l in envp.read_text(encoding="utf-8").splitlines():
        if l.startswith("HF_TOKEN="):
            tok = l.split("=", 1)[1].strip()

    download_and_verify(tok)
