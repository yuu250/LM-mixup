#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

MODEL_PATH = "BAAI/bge-m3"
DEVICE = "cuda"
BATCH_SIZE = 16
NORMALIZE = True
METRIC = "cosine"    
K_T = 3              
OUT_ROOT = Path("EasyR1/emb_lib")

SUBDIRS = {
    "qa": "qa",
    "mcq": "mcq",
    "tfq": "tfq",
    "paragraph": "paragraph",
    "cs": "cs",
}

random.seed(42)
np.random.seed(42)

def read_json_any(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".jsonl":
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in ["data", "items", "rows"]:
                if k in data and isinstance(data[k], list):
                    return data[k]
            return [data]
    raise ValueError(f"unsupported file type: {p.suffix}")

def get_by_dotted_key(obj: Dict[str, Any], dotted: str, default=None):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def coerce_score_to_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float):
            return int(round(v))
        if isinstance(v, str) and v.strip() != "":
            fv = float(v.strip())
            return int(round(fv))
    except Exception:
        return None
    return None

ANSWER_TAIL_RE = re.compile(r"\n\s*Answer\s*:\s*[^\n]*\s*$", re.IGNORECASE)

def strip_answer_tail(text: str) -> str:
    return ANSWER_TAIL_RE.sub("", text or "")

def build_text_qa(r: Dict[str, Any]) -> Optional[str]:
    q = (r.get("question") or "").strip()
    a = (r.get("answer") or "").strip()
    if not q:
        return None
    return f"Q: {q}\nA: {a}".strip()

def build_text_mcq(r: Dict[str, Any]) -> Optional[str]:
    raw = (r.get("fine_mcq") or "").strip()
    return strip_answer_tail(raw).strip() if raw else None

def build_text_tfq(r: Dict[str, Any]) -> Optional[str]:
    raw = (r.get("tfq_part") or "").strip()
    return strip_answer_tail(raw).strip() if raw else None

def build_text_paragraph(r: Dict[str, Any]) -> Optional[str]:
    para = (r.get("paragraph") or "").strip()
    return para or None

def build_text_cs(r: Dict[str, Any]) -> Optional[str]:
    cat  = (r.get("category") or "").strip()
    summ = (r.get("summarised_text") or "").strip()
    if not cat and not summ:
        return None
    return f"C: {cat}\nS: {summ}".strip()

BUILDERS = {
    "qa":        build_text_qa,
    "mcq":       build_text_mcq,
    "tfq":       build_text_tfq,
    "paragraph": build_text_paragraph,
    "cs":        build_text_cs,
}

_model_cache = {}
def get_embedder(model_path: str, device: str):
    key = (model_path, device)
    if key not in _model_cache:
        _model_cache[key] = SentenceTransformer(model_path, device=device)
    return _model_cache[key]

def encode_texts(texts: List[str], model_path: str, device: str,
                 batch_size: int, normalize: bool) -> np.ndarray:
    model = get_embedder(model_path, device)
    emb = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=normalize
    )
    return emb.astype(np.float32)

class DiscreteIndexer:
    def __init__(self, values: np.ndarray):
        vals = np.sort(np.unique(values.astype(int)))
        self.values = vals
        self.val2idx = {int(v): i for i, v in enumerate(vals)}
    @property
    def n_classes(self): return len(self.values)
    def to_index(self, arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr, dtype=int)
        for i, v in enumerate(arr.astype(int)):
            out[i] = self.val2idx[int(v)]
        return out
    def to_value_expectation(self, post: np.ndarray) -> float:
        return float(np.dot(self.values.astype(float), post))

def estimate_T_from_B(B_emb: np.ndarray,
                      B_scores_idx: np.ndarray,
                      n_classes: int,
                      k_t: int,
                      metric: str,
                      smoothing: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=min(k_t+1, len(B_emb)), metric=metric)
    nn.fit(B_emb)
    _, idx = nn.kneighbors(B_emb, return_distance=True)

    C = np.full((n_classes, n_classes), smoothing, dtype=float)
    for center, neighs in enumerate(idx):
        ci = int(B_scores_idx[center])
        for nb in neighs[1:1+k_t]:
            cj = int(B_scores_idx[nb])
            C[ci, cj] += 1.0

    T = C / C.sum(axis=1, keepdims=True)
    counts = np.bincount(B_scores_idx, minlength=n_classes).astype(float) + 1e-8
    p = counts / counts.sum()
    return T, p

def build_scored_lib(task: str, in_path: Optional[str], out_root: Path,
                     score_key: str,
                     model_path: str, device: str,
                     batch_size: int, normalize: bool,
                     metric: str, k_t: int):
    if not in_path:
        print(f"[SKIP] {task}: no input path")
        return

    rows = read_json_any(in_path)
    builder = BUILDERS[task]

    texts, uids, scores = [], [], []
    skipped_no_text, skipped_no_score = 0, 0

    for i, r in enumerate(rows):
        t = builder(r)
        if not t:
            skipped_no_text += 1
            continue
        sc = get_by_dotted_key(r, score_key, None)
        sc = coerce_score_to_int(sc)
        if sc is None:
            skipped_no_score += 1
            continue
        texts.append(t)
        uids.append(r.get("id", i))
        scores.append(int(sc))

    if not texts:
        print(f"[WARN] {task}: no valid items (no_text={skipped_no_text}, no_score={skipped_no_score})")
        return

    scores = np.array(scores, dtype=int)
    print(f"[INFO] {task}: parsed {len(texts)} items from {in_path} "
          f"(drop: no_text={skipped_no_text}, no_score={skipped_no_score}); "
          f"classes={np.unique(scores)}")

    emb = encode_texts(texts, model_path, device, batch_size, normalize)

    indexer = DiscreteIndexer(np.unique(scores))
    B_idx = indexer.to_index(scores)
    T, p = estimate_T_from_B(emb, B_idx, indexer.n_classes, k_t=k_t, metric=metric)

    out_dir = out_root / SUBDIRS[task]
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "B_emb.npy", emb)
    np.save(out_dir / "B_scores.npy", scores)

    nn_index = NearestNeighbors(n_neighbors=min(16, len(emb)), metric=metric, algorithm='brute')
    nn_index.fit(emb)
    joblib.dump(nn_index, out_dir / "nn_index.pkl")

    meta = {
        "task": task,
        "model_path": model_path,
        "device_hint": device,
        "normalize_embeddings": normalize,
        "batch_size": batch_size,
        "metric": metric,
        "k_t": int(k_t),
        "n_items": int(len(texts)),
        "source_path": str(in_path),
        "class_values": indexer.values.astype(int).tolist(),
        "T": T.tolist(),
        "p": p.tolist(),
        "k_default": 16
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] {task}: saved to {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Build scored embedding libraries for multiple tasks (Bayes-ready)")
    ap.add_argument("--qa_path",        type=str, default=None)
    ap.add_argument("--mcq_path",       type=str, default=None)
    ap.add_argument("--tfq_path",       type=str, default=None)
    ap.add_argument("--paragraphs_path",type=str, default=None)
    ap.add_argument("--cs_path",        type=str, default=None)
    ap.add_argument("--out_root",       type=str, default=str(OUT_ROOT))

    ap.add_argument("--score_key",      type=str, default="rating_detail.Overall_compressed")

    ap.add_argument("--model_path",     type=str, default=MODEL_PATH)
    ap.add_argument("--device",         type=str, default=DEVICE)
    ap.add_argument("--batch_size",     type=int, default=BATCH_SIZE)
    ap.add_argument("--normalize",      action="store_true", help="L2 normalize embeddings")
    ap.add_argument("--no-normalize",   dest="normalize", action="store_false")
    ap.set_defaults(normalize=NORMALIZE)
    ap.add_argument("--metric",         type=str, default=METRIC, choices=["cosine", "euclidean", "manhattan"])
    ap.add_argument("--k_t",            type=int, default=K_T)

    args = ap.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    build_scored_lib("qa",        args.qa_path,        out_root, args.score_key,
                     args.model_path, args.device, args.batch_size, args.normalize,
                     args.metric, args.k_t)
    build_scored_lib("mcq",       args.mcq_path,       out_root, args.score_key,
                     args.model_path, args.device, args.batch_size, args.normalize,
                     args.metric, args.k_t)
    build_scored_lib("tfq",       args.tfq_path,       out_root, args.score_key,
                     args.model_path, args.device, args.batch_size, args.normalize,
                     args.metric, args.k_t)
    build_scored_lib("paragraph", args.paragraphs_path,out_root, args.score_key,
                     args.model_path, args.device, args.batch_size, args.normalize,
                     args.metric, args.k_t)
    build_scored_lib("cs",        args.cs_path,        out_root, args.score_key,
                     args.model_path, args.device, args.batch_size, args.normalize,
                     args.metric, args.k_t)

if __name__ == "__main__":
    main()


"""
python build.py \
  --qa_path qa.jsonl \
  --mcq_path mcq.jsonl \
  --tfq_path tfq.jsonl \
  --paragraphs_path paragraphs.jsonl \
  --cs_path cs.jsonl \
  --out_root EasyR1/emb_lib \
  --score_key rating_detail.Overall_compressed \
  --model_path BAAI/bge-m3 \
  --device cuda --batch_size 16 --normalize --metric cosine --k_t 2
"""