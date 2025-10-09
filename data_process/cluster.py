import os
import json
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

import matplotlib
matplotlib.use("Agg")  # safe for servers without display
import matplotlib.pyplot as plt


def derive_dataset_from_path(input_path: str) -> str:
    name = os.path.basename(input_path)
    stem = os.path.splitext(name)[0]  # e.g., 'dolly_low'
    if stem.endswith("_low"):
        ds = stem[:-4]
        return ds if ds else stem
    return stem


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_jsonl(rows: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_text_prompt_answer(item: Dict[str, Any]) -> Optional[str]:
    prompt = item.get("prompt", "")
    answer = item.get("answer", item.get("completion", ""))
    text = f"{prompt}\n{answer}".strip()
    return text if text else None


def filter_low_quality(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        score = r.get("rating_detail", {}).get("Overall_compressed", None)
        if isinstance(score, int) and score < 4:
            out.append(r)
    return out

def encode_texts(
    texts: List[str],
    model_path: str,
    device: str = "cuda",
    batch_size: int = 8,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    model = SentenceTransformer(model_path, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return emb.astype(np.float32)

def sample_caps(
    N: int,
    min_sz: int = 2,
    max_sz: int = 25,
    mean: float = 10.0,
    std: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    caps = []
    total = 0
    while total < N:
        x = rng.normal(loc=mean, scale=std)
        x = int(np.clip(round(x), min_sz, max_sz))
        caps.append(x)
        total += x

    # Adjust tail to match sum==N
    excess = total - N
    if excess > 0:
        i = len(caps) - 1
        while excess > 0 and i >= 0:
            can_give = caps[i] - min_sz
            give = min(can_give, excess)
            caps[i] -= give
            excess -= give
            if caps[i] == 0:
                caps[i] = min_sz
            i -= 1
        if excess > 0:
            for i in range(len(caps)):
                can_give = caps[i] - min_sz
                if can_give <= 0:
                    continue
                give = min(can_give, excess)
                caps[i] -= give
                excess -= give
                if excess == 0:
                    break

    caps = [c for c in caps if c > 0]
    rng.shuffle(caps)
    caps = [int(np.clip(c, min_sz, max_sz)) for c in caps]

    diff = sum(caps) - N
    if diff != 0:
        step = 1 if diff < 0 else -1
        diff = abs(diff)
        idx = 0
        while diff > 0 and len(caps) > 0:
            j = idx % len(caps)
            if step == -1 and caps[j] > min_sz:
                caps[j] -= 1; diff -= 1
            elif step == 1 and caps[j] < max_sz:
                caps[j] += 1; diff -= 1
            idx += 1

    assert sum(caps) == N and all(min_sz <= c <= max_sz for c in caps), "Invalid capacity vector"
    return np.array(caps, dtype=np.int32)



def capacity_constrained_assign(
    X: np.ndarray,
    centers: np.ndarray,
    caps: np.ndarray,
    min_size: int = 2,
    device_t: str = "cuda",
    batch: int = 2048,
) -> np.ndarray:
    N = len(X)
    k = len(centers)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device_t)
    C = torch.from_numpy(centers.astype(np.float32)).to(device_t)

    labels = np.full(N, -1, dtype=np.int32)
    sizes = np.zeros(k, dtype=np.int32)

    with torch.no_grad():
        top1_scores = torch.empty(N, device=device_t)
        for s in range(0, N, batch):
            e = min(N, s + batch)
            sims = X_t[s:e] @ C.T
            top1_scores[s:e], _ = sims.max(dim=1)
    prio = torch.argsort(top1_scores, descending=True).cpu().numpy()

    with torch.no_grad():
        for off in range(0, N, batch):
            batch_idx = prio[off:off+batch]
            sims = X_t[batch_idx] @ C.T   # (B, k)
            order = torch.argsort(sims, dim=1, descending=True).cpu().numpy()
            for row, i in enumerate(batch_idx):
                for c in order[row]:
                    if sizes[c] < caps[c]:
                        labels[i] = c
                        sizes[c] += 1
                        break

    unassigned = np.where(labels == -1)[0]
    if len(unassigned) > 0:
        with torch.no_grad():
            sims = X_t[unassigned] @ C.T
            order = torch.argsort(sims, dim=1, descending=True).cpu().numpy()
        for r, i in enumerate(unassigned):
            for c in order[r]:
                if sizes[c] < caps[c]:
                    labels[i] = c
                    sizes[c] += 1
                    break

    changed = True
    while changed:
        changed = False
        small = np.where(sizes < min_size)[0]
        if len(small) == 0:
            break
        for c in small:
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                sizes[c] = 0
                continue
            with torch.no_grad():
                sims = X_t[idxs] @ C.T
                order = torch.argsort(sims, dim=1, descending=True).cpu().numpy()
            for r, i in enumerate(idxs):
                for c2 in order[r]:
                    if c2 != c and sizes[c2] < caps[c2]:
                        labels[i] = c2
                        sizes[c2] += 1
                        changed = True
                        break
            sizes[c] = np.sum(labels == c)

    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq)}
    labels = np.array([remap[x] for x in labels], dtype=np.int32)
    return labels


def transform_with_cluster(
    rows: List[Dict[str, Any]],
    texts: List[str],
    dataset: str,
    labels: np.ndarray,
) -> List[Dict[str, Any]]:
    out = []
    for i, (r, txt) in enumerate(zip(rows, texts)):
        rec = dict(r)
        rec.pop("cluster_label", None)  # drop any stale field
        rec.pop("id", None)             # ensure 'id' not present
        rec["dataset"] = dataset
        rec["text_used"] = txt
        rec["cluster_label"] = int(labels[i])
        out.append(rec)
    return out


def save_split_by_cluster(
    rows_transformed: List[Dict[str, Any]],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({"idx": np.arange(len(rows_transformed)),
                       "label": [r["cluster_label"] for r in rows_transformed]})
    for lab, g in df.groupby("label"):
        out_path = os.path.join(out_dir, f"cluster_{int(lab)}.jsonl")
        subset = [rows_transformed[i] for i in g["idx"].tolist()]
        save_jsonl(subset, out_path)


def save_cluster_size_plot(labels: np.ndarray, png_path: str):
    df = pd.DataFrame({"label": labels})
    sz = df.groupby("label").size()
    plt.figure(figsize=(8, 4))
    sz.value_counts().sort_index().plot(kind="bar", edgecolor="black")
    plt.xlabel("Cluster Size")
    plt.ylabel("Number of Clusters")
    plt.title("Cluster Size Distribution (capacity-constrained)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Input JSONL path")
    ap.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL (with cluster_label, no id)")
    ap.add_argument("--split_dir", type=str, default=None, help="Optional: directory to save per-cluster JSONL")
    ap.add_argument("--plot_png", type=str, default=None, help="Optional: save cluster-size histogram PNG")

    # embedding
    ap.add_argument("--model_path", type=str, default="BAAI/bge-m3")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--normalize_embeddings", action="store_true")
    ap.add_argument("--no-normalize_embeddings", dest="normalize_embeddings", action="store_false")
    ap.set_defaults(normalize_embeddings=True)

    # capacity settings
    ap.add_argument("--min_size", type=int, default=2)
    ap.add_argument("--max_size", type=int, default=25)
    ap.add_argument("--mean_cap", type=float, default=10.0)
    ap.add_argument("--std_cap", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)

    # kmeans params
    ap.add_argument("--kmeans_batch", type=int, default=2048)
    ap.add_argument("--kmeans_max_iter", type=int, default=30)

    args = ap.parse_args()

    # dataset name from input filename
    dataset = derive_dataset_from_path(args.input)
    print(f"[INFO] Parsed dataset name: '{dataset}' from '{args.input}'")

    # 1) load & filter low-quality
    rows_all = read_jsonl(args.input)
    # rows = filter_low_quality(rows_all)
    # if len(rows) == 0:
    #     raise ValueError("No low-quality items (Overall_compressed < 4) found in input.")
    rows = rows_all
    print(f"[INFO] Total items in input: {len(rows_all)}")

    # 2) build texts from prompt + answer
    texts, keep_rows = [], []
    for r in rows:
        txt = build_text_prompt_answer(r)
        if txt:
            texts.append(txt)
            keep_rows.append(r)
    rows = keep_rows
    if len(rows) == 0:
        raise ValueError("No valid items with (prompt + answer) available.")
    print(f"[INFO] Low-quality items after filtering: {len(rows)}")

    # 3) embedding
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    X = encode_texts(
        texts,
        model_path=args.model_path,
        device=device,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
    )
    N = len(X)

    # 4) capacity vector & k
    caps = sample_caps(
        N,
        min_sz=args.min_size,
        max_sz=args.max_size,
        mean=args.mean_cap,
        std=args.std_cap,
        seed=args.seed,
    )
    k = int(len(caps))
    print(f"[INFO] target k={k}, cap stats: min={caps.min()}, mean={caps.mean():.2f}, max={caps.max()}")

    # 5) initial centers by MiniBatchKMeans
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=args.seed,
        batch_size=args.kmeans_batch,
        max_iter=args.kmeans_max_iter,
        n_init=1
    )
    km.fit(X)
    centers = normalize(km.cluster_centers_.astype(np.float32), norm="l2", axis=1)
    Xn = X.astype(np.float32)  # already normalized if normalize_embeddings=True

    # 6) capacity-constrained assignment
    labels = capacity_constrained_assign(
        X=Xn,
        centers=centers,
        caps=caps,
        min_size=args.min_size,
        device_t=device,
        batch=2048
    )

    # 7) transform with cluster label (no id), and save (sorted by cluster_label)
    transformed_all = transform_with_cluster(rows, texts, dataset, labels)
    transformed_all.sort(key=lambda r: r["cluster_label"])
    save_jsonl(transformed_all, args.output_jsonl)
    print(f"[OK] JSONL (with cluster_label, grouped by cluster) written to: {args.output_jsonl}")

    if args.split_dir:
        save_split_by_cluster(transformed_all, args.split_dir)
        print(f"[OK] Per-cluster JSONL saved under: {args.split_dir}")

    if args.plot_png:
        save_cluster_size_plot(labels, args.plot_png)
        print(f"[OK] Cluster-size histogram written to: {args.plot_png}")

    # stats
    df = pd.DataFrame({"label": labels})
    sz = df.groupby("label").size()
    print("\n[STATS] cluster size describe:\n", sz.describe())
    print("\n[STATS] size -> #clusters\n", sz.value_counts().sort_index())


if __name__ == "__main__":
    main()