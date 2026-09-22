import json
import random
from pathlib import Path
from collections import defaultdict

def load_unified(path="data/processed/unified.jsonl"):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def stratified_split(records, val_frac=0.15, seed=42):
    random.seed(seed)
    # group by (tier, language) so each group gets proportional val split
    groups = defaultdict(list)
    for r in records:
        key = (r["tier"], r.get("language", "unknown"))
        groups[key].append(r)

    train, val = [], []
    for key, group in groups.items():
        random.shuffle(group)
        n_val = int(len(group) * val_frac)
        if n_val == 0 and len(group) >= 2:
            n_val = 1
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val

def write_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    records = load_unified()
    train, val = stratified_split(records)
    write_jsonl(train, "data/processed/train_split.jsonl")
    write_jsonl(val, "data/processed/val_split.jsonl")
    print(f"Train: {len(train)}, Val: {len(val)}")