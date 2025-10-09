import argparse
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List, Union

def iter_jsonl(fp: Path) -> Iterable[Dict[str, Any]]:
    with fp.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed line: {fp.name}#{ln}")
                continue

def iter_json(fp: Path) -> Iterable[Dict[str, Any]]:
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
            else:
                print("[WARN] Skipping non-dict item in JSON array")
    elif isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            for obj in data["data"]:
                if isinstance(obj, dict):
                    yield obj
                else:
                    print("[WARN] Skipping non-dict item in 'data' array")
        elif "items" in data and isinstance(data["items"], list):
            for obj in data["items"]:
                if isinstance(obj, dict):
                    yield obj
                else:
                    print("[WARN] Skipping non-dict item in 'items' array")
        else:
            yield data
    else:
        print("[WARN] Top-level JSON is neither an object nor an array")

def get_overall_compressed(rd: Dict[str, Any]) -> Union[int, float, None]:
    if not isinstance(rd, dict):
        return None
    return rd.get("Overall_compressed", rd.get("Overall_Compressed"))

def load_records(in_path: Path) -> Iterable[Dict[str, Any]]:
    if in_path.suffix.lower() == ".jsonl":
        return iter_jsonl(in_path)
    elif in_path.suffix.lower() == ".json":
        return iter_json(in_path)
    else:
        raise ValueError("Input file must have .json or .jsonl extension")

def write_output(objs: List[Dict[str, Any]], out_path: Path) -> None:
    if out_path.suffix.lower() == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for obj in objs:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif out_path.suffix.lower() == ".json":
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(objs, f, ensure_ascii=False)
    else:
        raise ValueError("Output file must have .json or .jsonl extension")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to the input JSON or JSONL file")
    ap.add_argument("--output", required=True, help="Path to the output JSON or JSONL file")
    ap.add_argument("--min-score", type=float, default=4.0, help="Minimum Overall_compressed score to keep")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    kept_objs: List[Dict[str, Any]] = []
    for obj in load_records(in_path):
        rd = obj.get("rating_detail", {})
        oc = get_overall_compressed(rd)
        if oc is not None and isinstance(oc, (int, float)) and oc >= args.min_score:
            kept_objs.append(obj)

    write_output(kept_objs, out_path)
    print(f"Done: kept {len(kept_objs)} records -> {out_path}")

if __name__ == "__main__":
    main()