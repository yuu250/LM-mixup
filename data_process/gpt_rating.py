import os, re, json, time, requests, concurrent.futures, regex
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm

def gpt_4_call(text: str, api_key: str, url: str,
               model: str = "gpt-4o-mini",
               retries: int = 5, delay: int = 25) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}]
    })
    headers = {
        "Accept": "application/json",
        "Authorization": api_key, 
        "Content-Type": "application/json"
    }
    for i in range(retries):
        try:
            r = requests.post(url, headers=headers, data=payload, timeout=20)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Retry {i+1}/{retries}] {e}")
            time.sleep(delay)
    return "exception"

json_obj_pat   = regex.compile(r"\{(?:[^{}]|(?R))*\}")
json_array_pat = regex.compile(r"\[(?:[^\[\]]|(?R))*\]")

def extract_scores(raw: str):
    txt = raw.strip()
    m_arr = json_array_pat.search(txt)
    m_obj = json_obj_pat.search(txt)
    try:
        objs = json.loads(m_arr.group() if m_arr else m_obj.group())
        if isinstance(objs, dict):
            objs = [objs]
    except Exception:
        return []
    res = []
    for o in objs:
        res.append([
            int(o.get("Rarity", 0)),
            int(o.get("Complexity", 0)),
            int(o.get("Informativeness", 0)),
            int(o.get("Overall rating", 0) or o.get("Overall", 0))
        ])
    return res


def compress_overall(vals):
    print("Original distribution :", Counter(vals))
    comp = [min(9, max(4, v)) - 4 for v in vals]   
    print("Compressed distribution:", Counter(comp))
    return comp

TASK_TYPES = ("qa", "mcq", "para", "tfq", "cs")

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def infer_task_type(rec: Dict[str, Any]) -> str:
    src = _norm(rec.get("source"))
    if src in TASK_TYPES:
        return src

    rid = _norm(str(rec.get("id", "")))
    if rid:
        tokens = re.split(r"[\s_\-:/\.]+", rid)
        for t in TASK_TYPES:
            if t in tokens:
                return t
        for t in TASK_TYPES:
            if re.search(fr"(?:^|[_\-\s]){t}(?:$|[_\-\s])", rid):
                return t
    return "qa"

def build_rubric(task: str) -> str:
    base = (
        "You are a data-quality estimator. "
        "Rate this {task_desc} on "
        "Rarity, Complexity, Informativeness, and Overall (1-10 integers).\n"
        "Return ONLY a JSON object like:\n"
        '{{ "Rarity": <int>, "Complexity": <int>, '
        '"Informativeness": <int>, "Overall rating": <int> }}\n\n'
        "### {task_name}:\n"
    )

    task_map = {
        "qa":   ("**single** QA pair", "QA Pair"),
        "mcq":  ("**single** MCQ item", "MCQ Item"),
        "para": ("**single** merged paragraph", "Paragraph"),
        "tfq":  ("**single** True/False question", "TFQ"),
        "cs":   ("**single** category–statement example", "CS Example"),
    }

    task_desc, task_name = task_map.get(task, ("**single** QA pair", "QA Pair"))
    return base.format(task_desc=task_desc, task_name=task_name)

def get_text_by_field(rec: Dict[str, Any], rating_field: str,
                      strict_field: bool = False) -> str:
    val = rec.get(rating_field, "")
    if isinstance(val, str) and val.strip():
        return val.strip()

    if not strict_field:
        for k in ("generated_output", "output"):
            v = rec.get(k, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

def main(input_jsonl      : str = "infer_out/qwe_1_5_high_qual.jsonl",
         output_dir       : str = "inter_out/rating",
         api_key          : str = "sk-xxx",
         url              : str = "https://api2.aigcbest.top/v1/chat/completions",
         model            : str = "gpt-4o-mini",
         max_workers      : int = 128,
         task_filter      : str = "qa",   # one of {"auto","qa","mcq","para","tfq","cs"}
         rating_field     : str = "generated_output",
         strict_field     : bool = True):

    records: List[Dict[str, Any]] = []
    in_path = Path(input_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found")

    if in_path.suffix.lower() == ".jsonl":
        with in_path.open("r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    else:
        with in_path.open("r", encoding="utf-8") as fr:
            data = json.load(fr)
        if isinstance(data, list):
            records = data
        else:
            raise ValueError("Input must be JSONL (one object per line) or a JSON array.")

    if not records:
        print("No valid lines found."); return

    task_filter = task_filter.strip().lower()
    if task_filter != "auto" and task_filter not in TASK_TYPES:
        raise ValueError(f"task_filter must be one of auto|{','.join(TASK_TYPES)}")

    prompts: List[str] = []
    tasks   : List[str] = []
    kept_records: List[Dict[str, Any]] = []

    for rec in records:
        task = infer_task_type(rec) if task_filter == "auto" else task_filter
        text_for_rating = get_text_by_field(rec, rating_field, strict_field=strict_field)
        if not text_for_rating:
            if strict_field:
                continue

        rubric = build_rubric(task)
        prompt = f"{rubric}\n\n### Output:\n{text_for_rating}\n\n### Rating:"
        prompts.append(prompt)
        tasks.append(task)
        kept_records.append(rec)

    if not kept_records:
        print("No records left for scoring (maybe strict_field=True and field missing).")
        return

    scores_per_rec = [[] for _ in kept_records]

    def _worker(idx, prompt):
        return idx, gpt_4_call(prompt, api_key, url, model=model)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_worker, i, p): i for i, p in enumerate(prompts)}
        for fut in tqdm(concurrent.futures.as_completed(futs),
                        total=len(futs), desc="Scoring"):
            i, raw = fut.result()
            scores = extract_scores(raw) or [[0, 0, 0, 0]]
            scores_per_rec[i] = scores[0]

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"scored_{in_path.name}")

    overall_vals = [sc[3] for sc in scores_per_rec]
    overall_compressed = compress_overall(overall_vals)

    per_type_raw   = defaultdict(list)   # 1–10
    per_type_comp  = defaultdict(list)   # 0–5

    with open(out_path, "w", encoding="utf-8") as fw:
        for rec, sc, comp, task in zip(kept_records, scores_per_rec, overall_compressed, tasks):
            rec["rating_detail"] = {
                "Rarity": sc[0],
                "Complexity": sc[1],
                "Informativeness": sc[2],
                "Overall": sc[3],
                "Overall_compressed": comp
            }
            rec["rated_task_type"] = task
            rec["__rating_field__"] = rating_field
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_type_raw[task].append(sc[3])
            per_type_comp[task].append(comp)

    raw_mean = float(np.mean(overall_vals)) if overall_vals else 0.0
    cmp_mean = float(np.mean(overall_compressed)) if overall_compressed else 0.0

    summary_lines = [
        "==== SUMMARY ====",
        f"Total items scored              : {len(overall_vals)}",
        f"Global Overall (raw, 1-10)      : {raw_mean:.2f}",
        f"Global Overall (compressed 0-5) : {cmp_mean:.2f}",
        f"Task mode                       : {task_filter}",
        f"Rating field                    : {rating_field} (strict={strict_field})",
        "",
        "---- Per-type Averages ----",
    ]
    for t in sorted(per_type_raw.keys()):
        r = per_type_raw[t]
        c = per_type_comp[t]
        if r and c:
            summary_lines.append(
                f"{t:>4} | n={len(r):4d} | raw={np.mean(r):.2f} | compressed={np.mean(c):.2f}"
            )
    summary_lines.append("==================")

    summary_txt = "\n".join(summary_lines)
    print(summary_txt)

    with open(os.path.join(output_dir, f"summary_{in_path.stem}.txt"),
              "w", encoding="utf-8") as sf:
        sf.write(summary_txt)
    print(f"★ Scored JSONL → {out_path}")

if __name__ == "__main__":
    import fire
    fire.Fire(main)


"""
python rating_anything.py \
    --input_jsonl="infer_out/qwe_7b/qwe_7b_dolly_low.jsonl" \
    --output_dir="infer_out/qwe_7b/rating"

python rating_anything.py \
    --input_jsonl="infer_out/qwe_7b/qwe_7b_oasst1_low.jsonl" \
    --output_dir="infer_out/qwe_7b/rating" 

python rating_anything.py \
    --input_jsonl="infer_out/qwe_7b/qwe_7b_stanford_alpaca_low.jsonl" \
    --output_dir="infer_out/qwe_7b/rating" 

python rating_anything.py \
    --input_jsonl="infer_out/qwe_7b/qwe_7b_wizardlm_low.jsonl" \
    --output_dir="infer_out/qwe_7b/rating" 

python rating_anything.py \
    --input_jsonl="infer_out/qwe_7b/qwe_7b_flan_v2.jsonl" \
    --output_dir="infer_out/qwe_7b/rating" 
"""