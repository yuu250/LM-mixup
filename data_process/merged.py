import argparse
import json
import random

def read_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def get_long_tail_norm(x):
    v = x.get("long_tail_norm", float("-inf"))
    try:
        return float(v)
    except Exception:
        return float("-inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="Path to the original high-quality JSONL file")
    ap.add_argument("--mixup", required=True, help="Path to the mixup JSONL file")
    ap.add_argument("--out", required=True, help="Path to the output JSONL file")
    ap.add_argument("--top-n", type=int, default=5000, help="Number of mixup samples")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = ap.parse_args()

    random.seed(args.seed)
    original = read_jsonl(args.original)
    mixup = read_jsonl(args.mixup)

    mixup_sorted = sorted(mixup, key=get_long_tail_norm, reverse=True)
    top_n = mixup_sorted[: args.top_n] if args.top_n <= len(mixup_sorted) else mixup_sorted

    merged = original + top_n
    filtered = [{"prompt": it.get("prompt", ""), "completion": it.get("completion", "")} for it in merged]
    random.shuffle(filtered)

    with open(args.out, "w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Selected {len(top_n)} from mixup (by long_tail_norm desc). "
          f"Merged total: {len(filtered)}. Saved to: {args.out}")

if __name__ == "__main__":
    main()