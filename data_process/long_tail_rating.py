import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


def read_jsonl(fp: Path):
    with fp.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] skip broken line: {fp}#{ln}")


def write_jsonl(fp: Path, rows: List[Dict[str, Any]]):
    with fp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dotted_get(d: Dict[str, Any], dotted_key: str, default=None):
    cur = d
    for k in dotted_key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_list(x):
    return x if isinstance(x, list) else [x]


def batched(iterable, n: int):
    batch = []
    for it in iterable:
        batch.append(it)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def load_model(model_name_or_path: str) -> SentenceTransformer:
    print(f"[INFO] loading model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path, device="cuda" if torch_cuda_available() else "cpu")
    return model


def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    embs: List[np.ndarray] = []
    for batch in tqdm(batched(texts, batch_size), total=(len(texts) + batch_size - 1) // batch_size, desc="Embedding"):
        arr = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embs.append(arr.astype(np.float32, copy=False))
    X = np.vstack(embs)
    assert X.shape[0] == len(texts)
    return X


def long_tail_knn_cosine(X: np.ndarray, k: int = 10) -> np.ndarray:
    k_eff = min(k + 1, X.shape[0])  # +1 for self, then drop self
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine", algorithm="auto", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, n_neighbors=k_eff, return_distance=True)
    # drop self at position 0
    mean_dist = distances[:, 1:].mean(axis=1) if k_eff > 1 else np.zeros(X.shape[0], dtype=np.float32)
    return mean_dist.astype(np.float32, copy=False)


def build_text(prompt: str, completion: str, joiner_prompt: str, joiner_answer: str) -> str:
    p = prompt if prompt is not None else ""
    c = completion if completion is not None else ""
    return f"{joiner_prompt}{p}\n{joiner_answer}{c}".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL with fields: prompt, completion")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL with long_tail_score etc.")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3",
                        help="bge-m3 path or name.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--knn_k", type=int, default=15)
    parser.add_argument("--normalize", action="store_true", default=True, help="Normalize embeddings (recommended)")
    parser.add_argument("--score_field", type=str, default="rating_detail.Overall_compressed",
                        help="Quality score dotted key. If not present, only long-tail will be used.")
    parser.add_argument("--min_quality", type=float, default=None,
                        help="Optional filter: keep items with quality >= min_quality before writing")
    parser.add_argument("--joiner_prompt", type=str, default="Instruction: ")
    parser.add_argument("--joiner_answer", type=str, default="Answer: ")
    args = parser.parse_args()
    print(f"[ARGS] {args}")
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = list(read_jsonl(in_path))
    if not rows:
        print(f"[ERROR] Empty input: {in_path}")
        return

    texts: List[str] = []
    for r in rows:
        prompt = r.get("prompt", "")
        completion = r.get("completion", "")
        texts.append(build_text(prompt, completion, args.joiner_prompt, args.joiner_answer))
    print(f"[INFO] read {len(rows)} rows from: {in_path}")
    model_name = args.model
    if model_name is None:
        local_try = None
        model_name = local_try if Path(local_try).exists() else "BAAI/bge-m3"
    model = load_model(model_name)
    print(f"[INFO] encoding {len(texts)} texts with batch_size={args.batch_size} ...")
    X = encode_texts(model, texts, batch_size=args.batch_size, normalize=args.normalize)

    lt = long_tail_knn_cosine(X, k=args.knn_k)  # bigger = more rare
    # Normalize long-tail to [0,1] for easier reading (optional)
    lt_min, lt_max = float(lt.min()), float(lt.max())
    if lt_max > lt_min:
        lt_norm = (lt - lt_min) / (lt_max - lt_min)
    else:
        lt_norm = np.zeros_like(lt)

    qualities: List[Optional[float]] = []
    for r in rows:
        q = dotted_get(r, args.score_field, default=None)
        try:
            q = float(q) if q is not None else None
        except Exception:
            q = None
        qualities.append(q)

    items: List[Tuple[int, Dict[str, Any]]] = []  # (original_index, row)
    for i, r in enumerate(rows):
        r = dict(r)  # shallow copy
        r["long_tail_score"] = float(lt[i])
        r["long_tail_norm"] = float(lt_norm[i])
        q = qualities[i]
        if q is not None:
            r["quality_score"] = float(q)
            # also provide 0..1 normalized quality if in 1..5
            if 1.0 <= q <= 5.0:
                r["quality_norm"] = (q - 1.0) / 4.0
        items.append((i, r))

    # Filter by min_quality if requested
    if args.min_quality is not None:
        kept = []
        for i, r in items:
            q = r.get("quality_score", None)
            if q is not None and q >= args.min_quality:
                kept.append((i, r))
        items = kept

    # Decide sorting rule
    has_quality = any(r.get("quality_score") is not None for _, r in items)
    if has_quality:
        items.sort(key=lambda x: (x[1].get("quality_score", -1e9), x[1]["long_tail_norm"]), reverse=True)
    else:
        items.sort(key=lambda x: x[1]["long_tail_norm"], reverse=True)

    # Assign final_rank and keep k info
    for rank, (_, r) in enumerate(items, start=1):
        r["final_rank"] = rank
        r["knn_k"] = args.knn_k

    out_rows = [r for _, r in items]
    write_jsonl(out_path, out_rows)
    print(f"[DONE] wrote {len(out_rows)} rows to: {out_path}")
    print(f" - used model: {model_name}")
    print(f" - knn_k={args.knn_k}, normalize={args.normalize}")
    if has_quality:
        print(f" - sorting by (quality desc, long_tail desc)")
    else:
        print(f" - sorting by (long_tail desc) only")
    if args.min_quality is not None:
        print(f" - filtered by min_quality >= {args.min_quality}")


if __name__ == "__main__":
    main()
