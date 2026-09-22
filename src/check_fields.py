import os
from datasets import load_dataset, Audio
from dotenv import load_dotenv

load_dotenv()
token = os.environ.get("HF_TOKEN")

dataset = load_dataset("ARTPARK-IISc/Vaani-Noise-Event-Dataset", token=token, streaming=True)
train_split = dataset["train"].cast_column("audio", Audio(decode=False))

sample = next(iter(train_split))
print(sample.keys())
print("annotationQuality:", sample.get("annotationQuality"))

seen = set()
for i, sample in enumerate(train_split):
    seen.add(sample.get("annotationQuality"))
    if len(seen) >= 5 or i > 2000:
        break
print(seen)