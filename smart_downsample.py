#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Smart downsampling with BGE-M3:
1) (optional) near-dup removal by cosine similarity threshold
2) kNN long-tail score (mean cosine distance to k neighbors)
3) stratified keep-top-pct by task_type (default 50%)

Usage:
  python smart_downsample.py \
    --in /root/autodl-tmp/EasyR1/data/llm_mixup/train.json \
    --out /root/autodl-tmp/EasyR1/data/llm_mixup/train_top50.json \
    --model /root/autodl-tmp/EasyR1/ckpt/bge-m3 \
    --k 16 --top_pct 0.5 --batch 64 \
    --dedup --dedup_thr 0.92
"""

import argparse, json, os, sys, math
from pathlib import Path
from typing import List, Dict, Any, Tuple
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


def embed_texts(texts: List[str], model_path: str, batch_size: int = 64, device: str = None) -> np.ndarray:
    model = SentenceTransformer(model_path, device=device or ("cuda" if _has_cuda() else "cpu"))
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # 归一化，cosine 相似度 = 点积
        show_progress_bar=True,
    )
    return embs.astype(np.float32)


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


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
        # 每个向量查询近邻（含自身），k 取一个较小上界（比如 50）做局部去重
        k = min(50, n)
        sims, idxs = index.search(embs, k)
        seen = np.zeros(n, dtype=bool)
        for i in range(n):
            if seen[i]:
                continue
            # 标记与 i 高相似度的邻居为“已见”
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

    # Fallback（无 faiss）：用简单分桶近似，速度不如 FAISS，适合中小规模
    # 将向量四舍五入到两位小数形成桶键，桶内逐一比对
    keys = np.round(embs, 2).astype(np.float32)
    # 哈希键：转 bytes
    buckets: Dict[bytes, List[int]] = {}
    for i in range(n):
        k = keys[i].tobytes()
        buckets.setdefault(k, []).append(i)

    # 桶内贪心过滤
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
    """
    使用 sklearn NearestNeighbors(metric='cosine') 计算 kNN 平均距离（排除自身）。
    因为已归一化，cosine 距离 = 1 - 相似度。
    """
    n = embs.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    k_eff = min(k + 1, n)  # +1 包含自身，之后再去掉
    nn = NearestNeighbors(n_neighbors=k_eff, metric="cosine", algorithm="auto")
    nn.fit(embs)
    dists, _ = nn.kneighbors(embs, return_distance=True)
    # dists[:, 0] 是自身距离=0，去掉
    if k_eff > 1:
        d = dists[:, 1:]  # shape (n, k)
    else:
        d = dists  # n==1 的边界
    # 均值即“长尾度”：越大越“远离众数”
    return d.mean(axis=1).astype(np.float32)


def stratified_top_pct(data: List[Dict[str, Any]], scores: np.ndarray, pct: float = 0.5) -> List[Dict[str, Any]]:
    """
    按 task_type 分层，保留各类分数 top pct。
    将分数写回每条记录的 'long_tail_score' 字段。
    """
    assert len(data) == len(scores)
    for rec, s in zip(data, scores):
        rec["long_tail_score"] = float(s)

    # 分组
    from collections import defaultdict
    groups = defaultdict(list)
    for rec in data:
        key = rec.get("task_type", "unknown")
        groups[key].append(rec)

    selected = []
    for key, items in groups.items():
        if not items:
            continue
        # 按分数降序
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
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    ap.add_argument("--k", type=int, default=16, help="kNN neighbors for long-tail score")
    ap.add_argument("--top_pct", type=float, default=0.5, help="Keep top pct per task_type")
    ap.add_argument("--dedup", action="store_true", help="Enable near-duplicate removal before scoring")
    ap.add_argument("--dedup_thr", type=float, default=0.92, help="Cosine similarity threshold for dedup (>=thr dropped)")
    args = ap.parse_args()

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

    # 嵌入
    print(f"[INFO] Encoding {len(texts)} samples with {args.model_path} ...")
    embs = embed_texts(texts, args.model_path, batch_size=args.batch)  # (N, D), L2 normalized

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

    # 打印各类统计
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

"""
python smart_downsample.py \
  --in  /root/autodl-tmp/EasyR1/data/llm_mixup/train.json \
  --out /root/autodl-tmp/EasyR1/data/llm_mixup/train_top50.json \
  --model /root/autodl-tmp/EasyR1/ckpt/bge-m3 \
  --k 16 --top_pct 0.5 --batch 64 \
  --dedup --dedup_thr 0.92
"""