# -*- coding: utf-8 -*-
import os
import re
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import regex
import joblib
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

JSON_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE
)
ANSWER_RE = re.compile(
    r"<answer>.*?Q:.*?A:.*?</answer>",
    re.DOTALL | re.IGNORECASE
)

@dataclass
class RewardWeights:
    quality: float = 0.6
    alignment: float = 0.3
    format: float = 0.1

    def normalize(self) -> "RewardWeights":
        s = self.quality + self.alignment + self.format
        if s <= 0:
            raise ValueError("Sum of weights must be > 0.")
        return RewardWeights(
            self.quality / s, self.alignment / s, self.format / s
        )


# ========== 贝叶斯评分系统核心组件 ==========
class DiscreteIndexer:
    """把离散分值（如 1..5）映射到 0..C-1 的索引，并支持反向期望计算"""
    def __init__(self, values: np.ndarray):
        vals = np.sort(np.unique(values.astype(int)))
        self.values = vals
        self.val2idx = {int(v): i for i, v in enumerate(vals)}

    @property
    def n_classes(self) -> int:
        return len(self.values)

    def to_index(self, arr: np.ndarray) -> np.ndarray:
        out = np.zeros_like(arr, dtype=int)
        for i, v in enumerate(arr.astype(int)):
            out[i] = self.val2idx[int(v)]
        return out

    def to_value_expectation(self, post: np.ndarray) -> float:
        return float(np.dot(self.values.astype(float), post))


def bayes_from_hist(neigh_hist: np.ndarray, T: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, float]:
    """log p_i + sum_j hist_j * log T_{i,j}  →  softmax → 后验"""
    logp = np.log(p + 1e-12)
    logT = np.log(T + 1e-12)
    C = T.shape[0]
    post_log = np.zeros(C, dtype=float)
    for i in range(C):
        post_log[i] = logp[i] + float(np.dot(neigh_hist, logT[i]))
    m = post_log.max()
    post = np.exp(post_log - m)
    post /= post.sum()
    entropy = float(-np.sum(post * np.log(post + 1e-12)))
    return post, entropy


def build_text(text_dict: Dict[str, Any]) -> str:
    """从字典构建文本，兼容不同的输入格式"""
    if isinstance(text_dict, str):
        return text_dict.strip()
    
    # 尝试提取不同的字段
    title = text_dict.get("title", "") or ""
    q = text_dict.get("question", "") or ""
    a = text_dict.get("answer", "") or ""
    
    # 如果有response字段，提取answer内容
    if "response" in text_dict:
        a = extract_answer_content(text_dict["response"])
    
    return f"{title}\nQ: {q}\nA: {a}".strip()


class BayesQualityScorer:
    """贝叶斯质量评分系统"""
    
    def __init__(self, lib_path: str, embedding_model_path: str, k_neighbors: int = 16):
        self.lib_path = Path(lib_path)
        self.k_neighbors = k_neighbors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = None
        self._load_assets(embedding_model_path)
    
    def _load_assets(self, embedding_model_path: str):
        """加载预训练的资产"""
        print(f"🔄 Loading Bayes scoring assets from {self.lib_path}")
        
        # 检查必要文件
        required_files = ["B_emb.npy", "B_scores.npy", "meta.json", "nn_index.pkl"]
        for file in required_files:
            if not (self.lib_path / file).exists():
                raise FileNotFoundError(f"Required asset file not found: {self.lib_path / file}")
        
        # 加载嵌入向量和评分
        self.B_emb = np.load(self.lib_path / "B_emb.npy")
        self.B_scores = np.load(self.lib_path / "B_scores.npy")
        
        # 加载元数据
        with open(self.lib_path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # 设置类别映射器
        class_values = np.array(meta["class_values"], dtype=int)
        self.indexer = DiscreteIndexer(class_values)
        self.B_idx = self.indexer.to_index(self.B_scores)
        
        # 加载转移矩阵和先验概率
        self.T = np.array(meta["T"], dtype=float)
        self.p = np.array(meta["p"], dtype=float)
        
        # 加载KNN索引
        self.nn = joblib.load(self.lib_path / "nn_index.pkl")
        
        # 加载嵌入模型
        print(f"🔄 Loading embedding model from {embedding_model_path} on device {self.device}")
        self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device)
        
        print("✅ Bayes scoring assets loaded successfully")
    
    def score_single(self, text: str) -> float:
        """对单个文本进行贝叶斯评分"""
        return self.score_batch([text])[0]
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """批量进行贝叶斯评分"""
        if not texts:
            return []
        
        # 向量化文本 - 与tmp.ipynb保持一致的参数
        X = self.embedding_model.encode(
            texts, 
            batch_size=16, 
            show_progress_bar=False,  # 训练时不显示进度条
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype(np.float32)
        
        # 批量检索近邻
        dists, idx = self.nn.kneighbors(X, n_neighbors=self.k_neighbors, return_distance=True)
        
        # 批量计算贝叶斯评分
        scores = []
        for row_i in idx:
            neigh_cls_idx = self.B_idx[row_i]
            hist = np.bincount(neigh_cls_idx, minlength=self.indexer.n_classes).astype(float)
            post, _ = bayes_from_hist(hist, self.T, self.p)
            
            # 使用连续期望作为评分 - 与tmp.ipynb一致，不做归一化
            pred_exp = self.indexer.to_value_expectation(post)
            
            # 将原始评分范围[0,5]映射到reward需要的[0,1]范围
            # 但保持与贝叶斯系统的原始逻辑一致
            #score = max(0.0, min(1.0, pred_exp / 5.0))
            if pred_exp >= 4:
                score = 1.0
            elif pred_exp == 3:
                score = 0.3
            else:
                score = 0.0
            scores.append(score)
        print(scores)
        return scores


def extract_answer_content(response: str) -> str:
    """Extract content from <answer>...</answer> tags"""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()




def format_reward(response: str) -> float:
    """Check if response matches the required format."""
    score = 0.0
    response = response or ""

    # 0.5 分：是否包裹在 <think> ... </think> 和 <answer> ... </answer> 中
    if re.fullmatch(FORMAT_RE, response):
        score += 0.5

    # 0.5 分：<answer> 内容中是否含有 Q: ... A: ...
    if re.search(ANSWER_RE, response):
        score += 0.5

    return score


class MLPClassifier(torch.nn.Module):
    """Simple MLP classifier for alignment scoring"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, x):
        return self.model(x)


def get_alignment_classifier(ckpt_path: str) -> MLPClassifier:
    """Load alignment classifier from checkpoint"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    if "input_dim" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain 'input_dim' and 'model_state_dict'.")

    input_dim = ckpt["input_dim"]
    state_dict = ckpt["model_state_dict"]

    model = MLPClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def alignment_reward(pair: str, alignment_clf, embedding_model) -> float:
    """Compute alignment reward using embedding + MLP classifier - single sample version"""
    return alignment_reward_batch([pair], alignment_clf, embedding_model)[0]


def alignment_reward_batch(pairs: List[str], alignment_clf, embedding_model) -> List[float]:
    """Compute alignment rewards for batch of text pairs using embedding + MLP classifier"""
    if not pairs:
        return []
    
    # 获取模型设备
    device = next(alignment_clf.parameters()).device
    
    # 批量嵌入
    features = embedding_model.encode(pairs, normalize_embeddings=True, convert_to_numpy=True)
    input_tensor = torch.tensor(features, dtype=torch.float32).to(device)  # 确保tensor在正确设备上
    
    # 批量预测
    with torch.no_grad():
        logits = alignment_clf(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1).cpu().numpy()
    
    # 批量转换为分数
    class_to_score = {0: 0.0, 1: 0.2, 2: 1.0}
    scores = [class_to_score.get(int(label), 0.0) for label in pred_labels]
    
    return scores


# Global variables to store models (avoiding Ray Actor complexity)
_alignment_clf = None
_embedding_model = None
_bayes_scorer = None
_weights = None
_initialized = False


def init_shared_reward_system(**kwargs):
    """Initialize the reward system in current process (no Ray Actor)"""
    global _alignment_clf, _embedding_model, _bayes_scorer, _weights, _initialized
    
    if _initialized:
        print(f"🔄 PID {os.getpid()}: Reward system already initialized, skipping")
        return
        
    print(f"🚀 PID {os.getpid()}: Initializing reward system...")
    print(f"Received kwargs: {kwargs}")
    
    try:
        # Initialize weights
        weights = kwargs.get('weights', RewardWeights())
        if hasattr(weights, '__dict__'):
            weights_dict = {'quality': weights.quality, 'alignment': weights.alignment, 'format': weights.format}
        else:
            weights_dict = weights
            
        total = weights_dict['quality'] + weights_dict['alignment'] + weights_dict['format']
        _weights = {
            'quality': weights_dict['quality'] / total,
            'alignment': weights_dict['alignment'] / total,
            'format': weights_dict['format'] / total
        }
        
        # 初始化贝叶斯评分器
        quality_config = kwargs.get('quality_scoring', {})
        knn_bayes_lib_path = kwargs.get('knn_bayes_lib_path', './knn_bayes_lib')
        embedding_model_path = kwargs.get('embedding_model_path', "/root/autodl-tmp/EasyR1/ckpt/bge-m3")
        k_neighbors = quality_config.get('k_neighbors', 16)
        
        print(f"🔄 Initializing Bayes quality scorer...")
        _bayes_scorer = BayesQualityScorer(knn_bayes_lib_path, embedding_model_path, k_neighbors)
        print("✅ Bayes quality scorer initialized")
        
        # Load alignment classifier
        alignment_model_path = kwargs.get('alignment_model_path', "/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt")
        print(f"Loading alignment classifier from {alignment_model_path}...")
        _alignment_clf = get_alignment_classifier(alignment_model_path)
        print("✅ Alignment classifier loaded")
        
        # Load embedding model for alignment scoring  
        embedding_model_path = kwargs.get('embedding_model_path', "/root/autodl-tmp/EasyR1/ckpt/bge-m3")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model for alignment scoring from {embedding_model_path} on device {device}...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(embedding_model_path, device=device)
        print("✅ Embedding model loaded")
        
        _initialized = True
        print("✅ Reward system initialization complete!")
        
    except Exception as e:
        print(f"❌ Reward system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    """
    批量计算reward score
    Args:
        reward_inputs: List of reward input dictionaries
        **kwargs: Additional arguments (e.g., format_weight)
    Returns:
        List of score dictionaries
    """
    global _alignment_clf, _embedding_model, _bayes_scorer, _weights, _initialized
    
    if not _initialized:
        raise RuntimeError("Reward system not initialized. Call init_shared_reward_system() first.")
    
    if not reward_inputs:
        return []
    
    # Normalize weights
    format_weight = kwargs.get('format_weight', 0.1)
    w_dict = {
        'quality': _weights['quality'],
        'alignment': _weights['alignment'], 
        'format': format_weight
    }
    total = w_dict['quality'] + w_dict['alignment'] + w_dict['format']
    w_normalized = {k: v/total for k, v in w_dict.items()}
    
    results = []
    
    # 批量处理：提取所有答案文本
    answer_texts = []
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        answer_part = extract_answer_content(response)
        answer_texts.append(answer_part)
    
    # 🚀 批量贝叶斯质量评分
    try:
        if _bayes_scorer is not None:
            q_scores = _bayes_scorer.score_batch(answer_texts)
            print(f"🔍 Batch Bayes quality scores computed for {len(answer_texts)} samples")
        else:
            raise RuntimeError("Bayes scorer not initialized")
    except Exception as e:
        print(f"⚠️ Batch Bayes quality scoring failed: {e}")
        # 如果贝叶斯评分失败，使用默认分数
        q_scores = [0.5] * len(answer_texts)
    
    # 🚀 批量对齐评分
    try:
        a_scores = alignment_reward_batch(answer_texts, _alignment_clf, _embedding_model)
        print(f"🔍 Batch alignment scores computed for {len(answer_texts)} samples")
    except Exception as e:
        print(f"⚠️ Batch alignment scoring failed: {e}")
        # 如果对齐评分失败，使用默认分数
        a_scores = [0.5] * len(answer_texts)
    
    # 逐个处理格式评分和最终组合
    for i, reward_input in enumerate(reward_inputs):
        response = reward_input.get("response", "")
        q_score = q_scores[i]
        a_score = a_scores[i]
        
        # Format score
        f_score = format_reward(response)
        if f_score < 1:
            print(f"格式分数低于1: {f_score}")
        
        overall = w_normalized['quality'] * q_score + w_normalized['format'] * f_score + w_normalized['alignment'] * a_score
        
        results.append({
            "overall": float(overall),
            "quality": float(q_score),
            "format": float(f_score),
            "alignment": float(a_score),
        })
    
    return results


# Usage example and testing
if __name__ == "__main__":
    # Initialize with Bayes scoring
    init_shared_reward_system(
        knn_bayes_lib_path="./knn_bayes_lib",
        alignment_model_path="/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt",
        embedding_model_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
        quality_scoring={
            "k_neighbors": 16,
            "enable_batch_scoring": True
        },
        weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
    )
    
    # Test samples
    samples = [
        {
            "response": "<think>2+2 is basic arithmetic</think><answer>Q: What is 2+2? A: 4</answer>",
            "question": "What is 2+2?",
            "answer": "4",
        },
        {
            "response": "<think>This is a complex mathematical problem that requires careful consideration of calculus</think><answer>Q: What is the derivative of x^2? A: The derivative of x^2 with respect to x is 2x, using the power rule</answer>",
            "question": "What is the derivative of x^2?",
            "answer": "2x",
        },
        {
            "response": "bad format without proper tags",
            "question": "Who wrote Hamlet?",
            "answer": "Shakespeare",
        }
    ]
    
    print(f"\n=== Testing Batch Scoring ===")
    scores = compute_score(samples)
    for i, (sample, score) in enumerate(zip(samples, scores)):
        print(f"\n=== Sample {i+1} ===")
        print(f"Response: {sample['response'][:60]}...")
        print(f"Scores: {score}")