import json
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

def load_unified(path="data/processed/unified.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def run_eda(records):
    durations = []
    for r in records:
        d = r.get("duration")
        if d is not None:
            try:
                durations.append(float(d))
            except (ValueError, TypeError):
                pass

    languages = Counter(r.get("language", "unknown") for r in records)
    tiers = Counter(r["tier"] for r in records)

    categories = Counter()
    for r in records:
        for ev in r.get("events", []):
            categories[ev.get("category", "unknown")] += 1
        for tag in r.get("noise_tags", []):
            # Cleanup tag format if it contains brackets/brackets
            clean_tag = tag.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
            categories[clean_tag] += 1

    Path("reports").mkdir(parents=True, exist_ok=True)

    print("=== Tier distribution ===")
    for k, v in tiers.items():
        print(f"{k}: {v}")

    print("\n=== Language distribution (top 15) ===")
    for lang, count in languages.most_common(15):
        print(f"{lang}: {count}")

    print("\n=== Noise category distribution (top 15) ===")
    for cat, count in categories.most_common(15):
        print(f"{cat}: {count}")

    if durations:
        print(f"\n=== Duration stats ===")
        print(f"count={len(durations)}, min={min(durations):.2f}, "
              f"max={max(durations):.2f}, mean={sum(durations)/len(durations):.2f}")

    # simple plots
    if durations:
        plt.figure()
        plt.hist(durations, bins=50)
        plt.title("Clip duration distribution")
        plt.xlabel("seconds")
        plt.savefig("reports/eda_duration_hist.png")
        plt.close()

    if languages:
        plt.figure(figsize=(10, 5))
        langs, counts = zip(*languages.most_common(15))
        plt.bar(langs, counts)
        plt.xticks(rotation=45, ha="right")
        plt.title("Language distribution")
        plt.tight_layout()
        plt.savefig("reports/eda_language_dist.png")
        plt.close()

    if categories:
        plt.figure(figsize=(10, 5))
        cats, counts = zip(*categories.most_common(15))
        plt.bar(cats, counts)
        plt.xticks(rotation=45, ha="right")
        plt.title("Noise category distribution")
        plt.tight_layout()
        plt.savefig("reports/eda_category_dist.png")
        plt.close()

if __name__ == "__main__":
    records = load_unified()
    run_eda(records)
