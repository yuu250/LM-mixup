#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smart downsampling with BGE-M3 (multi-GPU ready):
1) (optional) near-dup removal by cosine similarity threshold
2) kNN long-tail score (mean cosine distance to k neighbors)
3) stratified keep-top-pct by task_type (default 50%)

Usage:
  python smart_downsample_gpus.py \
    --in /root/autodl-tmp/EasyR1/data/llm_mixup/train.json \
    --out /root/autodl-tmp/EasyR1/data/llm_mixup/train_top35.json \
    --model /root/autodl-tmp/EasyR1/ckpt/bge-m3 \
    --k 16 --top_pct 0.35 --batch 64 \
    --devices 0,1,2 \
    --dedup --dedup_thr 0.92
"""

import argparse, json, os, sys, math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# --- Embedding ---
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("[ERR] Please install sentence-transformers: pip install sentence-transformers", file=sys.stderr)
    raise

# --- kNN ---
try:
    from sklearn.neighbors import NearestNeighbors
except Exception as e:
    print("[ERR] Please install scikit-learn: pip install scikit-learn", file=sys.stderr)
    raise

# (optional) FAISS for faster dedup
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def build_text(rec: Dict[str, Any]) -> str:
    # 支持 prompt/completion 或 input/output
    if "prompt" in rec or "completion" in rec:
        p = rec.get("prompt", "")
        c = rec.get("completion", "")
    else:
        p = rec.get("input", "")
        c = rec.get("output", "")
    return f"Instruction: {p}\nAnswer: {c}".strip()


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def parse_devices(devices_arg: str) -> List[str]:
    """
    解析 --devices 形如 "0,1,2" -> ["cuda:0","cuda:1","cuda:2"].
    为空则自动选择: 有GPU用cuda:0，否则cpu。
    """
    if not devices_arg:
        return ["cuda:0"] if _has_cuda() else ["cpu"]
    parts = [d.strip() for d in devices_arg.split(",") if d.strip() != ""]
    devs = [f"cuda:{p}" for p in parts]
    return devs or (["cuda:0"] if _has_cuda() else ["cpu"])


def embed_texts(texts: List[str], model_path: str, batch_size: int, devices: List[str]) -> np.ndarray:
    """
    多卡并行：当 devices>=2 时，使用 SentenceTransformer 的多进程多GPU池。
    单卡/CPU：常规 encode。
    """
    # 统一：多GPU时由主进程创建模型（不指定device），再启动多进程池
    if len(devices) >= 2 and all(d.startswith("cuda:") for d in devices):
        model = SentenceTransformer(model_path)  # device 由子进程各自设置
        print(f"[INFO] Using multi-GPU encoding on devices: {devices}")
        pool = model.start_multi_process_pool(target_devices=devices)
        embs = model.encode_multi_process(
            texts,
            pool,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        model.stop_multi_process_pool(pool)
    else:
        # 单设备（GPU 或 CPU）
        device = devices[0]
        print(f"[INFO] Using single device for encoding: {device}")
        model = SentenceTransformer(model_path, device=device)
        embs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    return np.asarray(embs, dtype=np.float32)


def dedup_by_cosine_threshold(embs: np.ndarray, thr: float = 0.92) -> np.ndarray:
    """
    返回保留的行索引（布尔掩码）。
    若安装了 faiss：用 IndexFlatIP 加速（相似度=点积；已归一化）。
    否则用一个简易“分桶+贪心”近似去重。
    """
    n = embs.shape[0]
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep

    if HAS_FAISS:
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        k = min(50, n)  # 每个向量查询的邻居数（含自身）
        sims, idxs = index.search(embs, k)
        seen = np.zeros(n, dtype=bool)
        for i in range(n):
            if seen[i]:
                continue
            nbrs = idxs[i]
            nbrs_s = sims[i]
            for j, s in zip(nbrs, nbrs_s):
                if j == i:
                    continue
                if s >= thr:
                    seen[j] = True
                    keep[j] = False
            seen[i] = True
        return keep

    # Fallback：简易分桶近似
    keys = np.round(embs, 2).astype(np.float32)
    buckets: Dict[bytes, List[int]] = {}
    for i in range(n):
        k = keys[i].tobytes()
        buckets.setdefault(k, []).append(i)

    for _, idx_list in buckets.items():
        if len(idx_list) <= 1:
            continue
        base = idx_list[0]
        for j in idx_list[1:]:
            if keep[j]:
                s = float(np.dot(embs[base], embs[j]))
                if s >= thr:
                    keep[j] = False
    return keep


def long_tail_score(embs: np.ndarray, k: int = 16) -> np.ndarray:
    """kNN 平均 cosine 距离（排除自身）。"""
    n = embs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine", algorithm="auto")
    nn.fit(embs)
    dists, _ = nn.kneighbors(embs, return_distance=True)
    d = dists[:, 1:] if k_eff > 1 else dists
    return d.mean(axis=1).astype(np.float32)


def stratified_top_pct(data: List[Dict[str, Any]], scores: np.ndarray, pct: float = 0.5) -> List[Dict[str, Any]]:
    """按 task_type 分层，保留各类分数 top pct；把分数写回 'long_tail_score'。"""
    assert len(data) == len(scores)
    for rec, s in zip(data, scores):
        rec["long_tail_score"] = float(s)

    from collections import defaultdict
    groups = defaultdict(list)
    for rec in data:
        key = rec.get("task_type", "unknown")
        groups[key].append(rec)

    selected = []
    for key, items in groups.items():
        if not items:
            continue
        items.sort(key=lambda r: r.get("long_tail_score", 0.0), reverse=True)
        m = len(items)
        keep_n = max(1, int(math.ceil(m * pct)))
        selected.extend(items[:keep_n])
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON file (list of dicts)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSON file (selected)")
    ap.add_argument("--model", dest="model_path", required=True, help="BGE-M3 model path")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size per device")
    ap.add_argument("--k", type=int, default=16, help="kNN neighbors for long-tail score")
    ap.add_argument("--top_pct", type=float, default=0.5, help="Keep top pct per task_type")
    ap.add_argument("--devices", type=str, default="", help='GPU id list like "0,1,2"; empty=auto')
    ap.add_argument("--dedup", action="store_true", help="Enable near-duplicate removal before scoring")
    ap.add_argument("--dedup_thr", type=float, default=0.92, help="Cosine similarity threshold for dedup (>=thr dropped)")
    args = ap.parse_args()

    devices = parse_devices(args.devices)
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("[ERR] Input JSON must be a list of records.", file=sys.stderr)
        sys.exit(1)

    # 文本构建
    texts = [build_text(rec) for rec in data]

    # 嵌入（多卡/单卡自适应）
    print(f"[INFO] Encoding {len(texts)} samples with {args.model_path} on devices={devices} ...")
    embs = embed_texts(texts, args.model_path, batch_size=args.batch, devices=devices)  # (N, D), L2 normalized

    # 去重（可选）
    if args.dedup:
        print(f"[INFO] Deduplicating with thr={args.dedup_thr} ...")
        keep_mask = dedup_by_cosine_threshold(embs, thr=args.dedup_thr)
        kept_idx = np.where(keep_mask)[0]
        print(f"[INFO] Keep {kept_idx.size}/{len(embs)} after dedup.")
        embs = embs[keep_mask]
        data = [data[i] for i in kept_idx]

    # 长尾分
    print(f"[INFO] Computing long-tail scores with k={args.k} ...")
    scores = long_tail_score(embs, k=args.k)

    # 分层取 top pct
    print(f"[INFO] Stratified selection: top {int(args.top_pct*100)}% per task_type ...")
    selected = stratified_top_pct(data, scores, pct=args.top_pct)

    # 写出
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {len(selected)} samples to {out_path}")

    # 统计
    from collections import Counter
    def counter(lst): return dict(Counter([r.get("task_type", "unknown") for r in lst]))
    before = counter(data)
    after = counter(selected)
    print("[STATS] per task_type (after / before):")
    keys = sorted(set(before) | set(after))
    for k in keys:
        print(f"  {k}: {after.get(k,0)} / {before.get(k,0)}")


if __name__ == "__main__":
    main()