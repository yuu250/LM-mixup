import json
import argparse
from collections import defaultdict

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # 跳过坏行
                continue
    return rows

def qa_to_pair(prompt: str, answer: str) -> str:
    p = (prompt or "").strip()
    a = (answer or "").strip()
    p = p.replace("\r\n", "\n").replace("\r", "\n").strip()
    a = a.replace("\r\n", "\n").replace("\r", "\n").strip()
    return f"Q: {p}\n A: {a}"

def group_and_merge(rows, sep=" "):
    buckets = defaultdict(list)
    dataset_name = None

    for r in rows:
        if dataset_name is None:
            dataset_name = r.get("dataset", None)
        cl = r.get("cluster_label", None)
        if cl is None:
            continue
        prompt = r.get("prompt", "")
        answer = r.get("completion", "")
        pair = qa_to_pair(prompt, answer)
        if pair.strip():
            buckets[cl].append(pair)

    merged = []
    for cl, pairs in sorted(buckets.items(), key=lambda kv: kv[0]):
        merged_input = sep.join(pairs) 
        rec = {
            "dataset": dataset_name if dataset_name is not None else rows[0].get("dataset", ""),
            "input": merged_input,
            "cluster_label": cl,
            "instruction": "Given some related question-answer pairs, merge them into a single coherent question-answer pair that preserves all essential information from the originals."
        }
        merged.append(rec)
    return merged

def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the input JSONL file")
    ap.add_argument("--output", required=True, help="Path to the output JSONL file (one line per cluster)")
    ap.add_argument("--sep", default=" ", help="Separator used to join Q/A segments within the same cluster; default is a space. You can change it to '\\n\\n'")
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    merged = group_and_merge(rows, sep=args.sep)
    write_jsonl(merged, args.output)
    print(f"[OK] Wrote {len(merged)} merged clusters to {args.output}")

if __name__ == "__main__":
    main()

"""
python cluter2input.py \
  --input tmp_merged/dolly_low.jsonl \
  --output infer_in/dolly_low.jsonl \
  --sep "\n\n"

python cluter2input.py \
  --input tmp_merged/stanford_alpaca_low.jsonl \
  --output infer_in/stanford_alpaca_low.jsonl \
  --sep "\n\n"

python cluter2input.py \
  --input tmp_merged/oasst1_low.jsonl \
  --output infer_in/oasst1_low.jsonl \
  --sep "\n\n"

python cluter2input.py \
  --input tmp_merged/wizardlm_low.jsonl \
  --output infer_in/wizardlm_low.jsonl \
  --sep "\n\n"

python cluter2input.py \
  --input tmp_merged/high_qual_clustered.jsonl \
  --output infer_in/high_qual_clustered.jsonl \
  --sep "\n\n"
"""
