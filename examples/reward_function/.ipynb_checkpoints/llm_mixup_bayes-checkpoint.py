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

# Task-specific format regex patterns
FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE
)

# Task-specific answer format patterns
ANSWER_PATTERNS = {
    "qa_mixup": re.compile(
        r"<answer>.*?Q:.*?A:.*?</answer>",
        re.DOTALL | re.IGNORECASE
    ),
    "mcq_mixup": re.compile(
        r"<answer>.*?Q:.*?A\).*?B\).*?C\).*?D\).*?Answer:.*?</answer>",
        re.DOTALL | re.IGNORECASE
    ),
    "cs_mixup": re.compile(
        r"<answer>.*?C:.*?S:.*?</answer>",
        re.DOTALL | re.IGNORECASE
    ),
    "tfq_mixup": re.compile(
        r"<answer>.*?Answer:.*?</answer>",
        re.DOTALL | re.IGNORECASE
    ),
    "para_mixup": None  # Only uses FORMAT_RE
}

@dataclass
class RewardWeights:
    quality: float = 0.7
    alignment: float = 0.2
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
    """Multi-task Bayes quality scoring system"""
    
    def __init__(self, base_lib_path: str, embedding_model_path: str, k_neighbors: int = 16):
        self.base_lib_path = Path(base_lib_path)
        self.k_neighbors = k_neighbors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = None
        self.task_scorers = {}
        self._load_embedding_model(embedding_model_path)
        self._load_task_assets()
    
    def _load_embedding_model(self, embedding_model_path: str):
        """Load embedding model"""
        print(f"🔄 Loading embedding model from {embedding_model_path} on device {self.device}")
        self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device)
        print("✅ Embedding model loaded successfully")
    
    def _load_task_assets(self):
        """Load assets for all supported tasks"""
        supported_tasks = ["qa", "mcq", "cs", "paragraph", "tfq"]
        
        for task in supported_tasks:
            task_path = self.base_lib_path / task
            if task_path.exists():
                try:
                    self._load_single_task_assets(task, task_path)
                    print(f"✅ Loaded assets for task: {task}")
                except Exception as e:
                    print(f"⚠️ Failed to load assets for task {task}: {e}")
            else:
                print(f"⚠️ Task directory not found: {task_path}")
    
    def _load_single_task_assets(self, task_name: str, task_path: Path):
        """Load assets for a single task"""
        # Check required files - consistent with qa_mixup_bayes.py
        required_files = ["B_emb.npy", "B_scores.npy", "meta.json", "nn_index.pkl"]
        for file in required_files:
            if not (task_path / file).exists():
                raise FileNotFoundError(f"Required file not found: {task_path / file}")
        
        # Load embeddings and scores - consistent with qa_mixup_bayes.py
        emb = np.load(task_path / "B_emb.npy")
        scores = np.load(task_path / "B_scores.npy")
        
        # Load metadata
        with open(task_path / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # Set up indexer - consistent with qa_mixup_bayes.py
        class_values = np.array(meta["class_values"], dtype=int)
        indexer = DiscreteIndexer(class_values)
        score_idx = indexer.to_index(scores)
        
        # Load transition matrix and prior - consistent with qa_mixup_bayes.py
        T = np.array(meta["T"], dtype=float)
        p = np.array(meta["p"], dtype=float)
        
        # Load pre-built KNN index - consistent with qa_mixup_bayes.py
        nn = joblib.load(task_path / "nn_index.pkl")
        
        # Store task scorer
        self.task_scorers[task_name] = {
            "emb": emb,
            "scores": scores,
            "indexer": indexer,
            "score_idx": score_idx,
            "T": T,
            "p": p,
            "nn": nn
        }
    
    def get_task_from_type(self, task_type: str) -> str:
        """Map task_type to internal task name"""
        type_mapping = {
            "qa_mixup": "qa",
            "mcq_mixup": "mcq", 
            "cs_mixup": "cs",
            "para_mixup": "paragraph",
            "tfq_mixup": "tfq"
        }
        return type_mapping.get(task_type, "qa")  # Default to qa
    
    def score_batch_by_task(self, texts: List[str], task_types: List[str]) -> List[float]:
        """Batch scoring with different tasks"""
        if not texts or not task_types:
            return []
        
        if len(texts) != len(task_types):
            raise ValueError("texts and task_types must have same length")
        
        # Group by task type
        task_groups = {}
        for i, task_type in enumerate(task_types):
            task = self.get_task_from_type(task_type)
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append((i, texts[i]))
        
        # Initialize results
        results = [0.0] * len(texts)
        
        # Process each task group
        for task, items in task_groups.items():
            if task not in self.task_scorers:
                print(f"⚠️ Task scorer not found for {task}, using default score 0.5")
                for idx, _ in items:
                    results[idx] = 0.5
                continue
            
            scorer = self.task_scorers[task]
            indices = [item[0] for item in items]
            task_texts = [item[1] for item in items]
            
            # Score this task group
            task_scores = self._score_single_task_batch(task_texts, scorer)
            
            # Put scores back in correct positions
            for i, score in enumerate(task_scores):
                results[indices[i]] = score
        
        return results
    
    def _score_single_task_batch(self, texts: List[str], scorer: Dict) -> List[float]:
        """Score batch for single task"""
        if not texts:
            return []
        
        # Encode texts
        X = self.embedding_model.encode(
            texts, 
            batch_size=16, 
            show_progress_bar=False,
            convert_to_numpy=True, 
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Find neighbors
        dists, idx = scorer["nn"].kneighbors(X, n_neighbors=self.k_neighbors, return_distance=True)
        
        # Compute Bayes scores
        scores = []
        for row_i in idx:
            neigh_cls_idx = scorer["score_idx"][row_i]
            hist = np.bincount(neigh_cls_idx, minlength=scorer["indexer"].n_classes).astype(float)
            post, _ = bayes_from_hist(hist, scorer["T"], scorer["p"])
            
            pred_exp = scorer["indexer"].to_value_expectation(post)
            
            # Map to reward range [0, 1]
            if pred_exp >= 4:
                score = 1.0
            elif pred_exp == 3:
                score = 0.3
            else:
                score = 0.0
            scores.append(score)
        
        return scores


def extract_answer_content(response: str) -> str:
    """Extract content from <answer>...</answer> tags"""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def format_reward(response: str, task_type: str = "qa_mixup") -> float:
    """Check if response matches the required format for specific task type."""
    score = 0.0
    response = response or ""

    # 0.5 points for proper <think>...</think> and <answer>...</answer> format
    if re.fullmatch(FORMAT_RE, response):
        score += 0.5

    # Task-specific answer format scoring
    if task_type == "para_mixup":
        # For paragraph mixup, only FORMAT_RE matters, so give full remaining score
        score += 0.5
    else:
        # For other tasks, check specific answer patterns
        answer_pattern = ANSWER_PATTERNS.get(task_type)
        if answer_pattern and re.search(answer_pattern, response):
            score += 0.5

    return score




def alignment_reward_batch(model_outputs: List[str], ground_truths: List[str], task_types: List[str], embedding_model) -> List[float]:
    """Compute alignment rewards using BGE-M3 similarity between model output and ground truth"""
    if not model_outputs or not ground_truths or not task_types:
        return []
    
    if len(model_outputs) != len(ground_truths) or len(model_outputs) != len(task_types):
        raise ValueError("model_outputs, ground_truths, and task_types must have same length")
    
    # Encode all texts in batch for efficiency
    all_texts = model_outputs + ground_truths
    embeddings = embedding_model.encode(
        all_texts, 
        batch_size=16, 
        show_progress_bar=False,
        convert_to_numpy=True, 
        normalize_embeddings=True
    ).astype(np.float32)
    
    # Split embeddings back into model outputs and ground truths
    n = len(model_outputs)
    output_embeddings = embeddings[:n]
    truth_embeddings = embeddings[n:]
    
    # Compute cosine similarities (embeddings are already normalized)
    similarities = np.sum(output_embeddings * truth_embeddings, axis=1)
    
    # Apply scoring rule: [0-0.75) -> 0, [0.75-1] -> 1
    scores = []
    for similarity in similarities:
        if similarity >= 0.8:
            scores.append(1.0)
        else:
            scores.append(0.0)
    
    return scores




# Global variables to store models
_embedding_model = None
_bayes_scorer = None
_weights = None
_initialized = False


def init_shared_reward_system(**kwargs):
    """Initialize the multi-task reward system"""
    global _embedding_model, _bayes_scorer, _weights, _initialized
    
    if _initialized:
        print(f"🔄 PID {os.getpid()}: Reward system already initialized, skipping")
        return
        
    print(f"🚀 PID {os.getpid()}: Initializing multi-task reward system...")
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
        
        # Initialize multi-task Bayes scorer
        quality_config = kwargs.get('quality_scoring', {})
        knn_bayes_lib_path = kwargs.get('knn_bayes_lib_path', '/root/autodl-tmp/EasyR1/emb_lib')
        embedding_model_path = kwargs.get('embedding_model_path', "/root/autodl-tmp/EasyR1/ckpt/bge-m3")
        k_neighbors = quality_config.get('k_neighbors', 16)
        
        print(f"🔄 Initializing multi-task Bayes quality scorer...")
        _bayes_scorer = BayesQualityScorer(knn_bayes_lib_path, embedding_model_path, k_neighbors)
        print("✅ Multi-task Bayes quality scorer initialized")
        
        # Load embedding model for alignment scoring (BGE-M3 similarity)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BGE-M3 embedding model for alignment scoring from {embedding_model_path} on device {device}...")
        _embedding_model = SentenceTransformer(embedding_model_path, device=device)
        print("✅ BGE-M3 embedding model loaded for alignment scoring")
        
        _initialized = True
        print("✅ Multi-task reward system initialization complete!")
        
    except Exception as e:
        print(f"❌ Reward system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def compute_score(reward_inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, float]]:
    """
    批量计算multi-task reward score
    Args:
        reward_inputs: List of reward input dictionaries (must include 'task_type' field)
        **kwargs: Additional arguments (e.g., format_weight)
    Returns:
        List of score dictionaries
    """
    global _embedding_model, _bayes_scorer, _weights, _initialized
    
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
    
    # Extract data for batch processing
    answer_texts = []
    ground_truths = []
    task_types = []
    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        answer_part = extract_answer_content(response)
        answer_texts.append(answer_part)
        
        # Extract ground truth answer
        ground_truth = reward_input.get("ground_truth", "")
        ground_truths.append(ground_truth)
        
        # Extract task_type (required field)
        task_type = reward_input.get("task_type", "qa_mixup")
        task_types.append(task_type)
    
    # 🚀 Batch quality scoring with task awareness
    try:
        if _bayes_scorer is not None:
            q_scores = _bayes_scorer.score_batch_by_task(answer_texts, task_types)
            print(f"🔍 Batch multi-task quality scores computed for {len(answer_texts)} samples")
            
            # Print quality scores for debugging
            print("📊 Quality Scores by Task:")
            task_score_groups = {}
            for i, (task_type, q_score) in enumerate(zip(task_types, q_scores)):
                if task_type not in task_score_groups:
                    task_score_groups[task_type] = []
                task_score_groups[task_type].append(q_score)
            
            for task_type, scores in task_score_groups.items():
                avg_score = sum(scores) / len(scores)
                print(f"  {task_type}: avg={avg_score:.3f}, count={len(scores)}, scores={[f'{s:.3f}' for s in scores[:5]]}")
            
        else:
            raise RuntimeError("Bayes scorer not initialized")
    except Exception as e:
        print(f"⚠️ Batch quality scoring failed: {e}")
        q_scores = [0.5] * len(answer_texts)
    
    # 🚀 Batch alignment scoring using BGE-M3 similarity
    try:
        a_scores = alignment_reward_batch(answer_texts, ground_truths, task_types, _embedding_model)
        print(f"🔍 Batch alignment scores computed for {len(answer_texts)} samples using BGE-M3 similarity")
    except Exception as e:
        print(f"⚠️ Batch alignment scoring failed: {e}")
        a_scores = [0.5] * len(answer_texts)
    
    # Get task-specific configuration
    task_specific_config = kwargs.get('task_specific_config', {})
    
    # Process format scoring and final combination
    for i, reward_input in enumerate(reward_inputs):
        response = reward_input.get("response", "")
        task_type = task_types[i]
        q_score = q_scores[i]
        a_score = a_scores[i]
        
        # Task-specific format score
        f_score = format_reward(response, task_type)
        if f_score < 1:
            print(f"格式分数低于1: {f_score} for task: {task_type}")
        
        # Get task-specific weights or use default
        if task_type in task_specific_config:
            task_config = task_specific_config[task_type]
            task_weights = {
                'quality': task_config.get('quality_weight_override') or w_normalized['quality'],
                'alignment': task_config.get('alignment_weight_override') or w_normalized['alignment'],
                'format': task_config.get('format_weight_override') or w_normalized['format'],
            }
        else:
            # Default weights for tasks not specified in config
            if task_type in ["qa_mixup", "cs_mixup"]:
                # QA and CS use standard weights (quality + alignment + format)
                task_weights = w_normalized
            else:
                # MCQ, TFQ, Para use quality-focused weights (no alignment)
                task_weights = {
                    'quality': 0.8,
                    'alignment': 0.0,
                    'format': 0.2
                }
        
        overall = task_weights['quality'] * q_score + task_weights['format'] * f_score + task_weights['alignment'] * a_score
        
        # Create result dictionary with only numeric values for metrics
        result = {
            "overall": float(overall),
            "quality": float(q_score),
            "format": float(f_score),
            "alignment": float(a_score),
        }
        
        # Add debug info if in debug mode (can be enabled via kwargs)
        debug_mode = kwargs.get('debug_mode', False)
        if debug_mode:
            print(f"Sample {i}: task_type={task_type}, weights={task_weights}, scores=q:{q_score:.3f} a:{a_score:.3f} f:{f_score:.3f} -> {overall:.3f}")
        
        results.append(result)
    
    return results


# Usage example and testing
if __name__ == "__main__":
    # Initialize with multi-task support (now uses BGE-M3 similarity for alignment)
    init_shared_reward_system(
        knn_bayes_lib_path="/root/autodl-tmp/EasyR1/emb_lib",
        embedding_model_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
        quality_scoring={
            "k_neighbors": 16,
            "enable_batch_scoring": True
        },
        weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
    )
    
    # Test samples with different task types
    samples = [
        {
            "response": "<think>2+2 is basic arithmetic</think><answer>Q: What is 2+2? A: 4</answer>",
            "question": "What is 2+2?",
            "answer": "4",
            "task_type": "qa_mixup",
        },
        {
            "response": "<think>This is a multiple choice question</think><answer>Q: What is the capital of France? A) London B) Berlin C) Paris D) Madrid Answer: C</answer>",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "task_type": "mcq_mixup",
        },
        {
            "response": "<think>This is a CS pair</think><answer>C: Write a function to add two numbers S: def add(a, b): return a + b</answer>",
            "question": "Write a function to add two numbers",
            "answer": "def add(a, b): return a + b",
            "task_type": "cs_mixup",
        },
        {
            "response": "<think>This is a TFQ question</think><answer>The capital of France is Paris. Answer: Paris</answer>",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "task_type": "tfq_mixup",
        },
        {
            "response": "<think>This is a paragraph mixup</think><answer>The Industrial Revolution was a period of major industrialization that began in Britain in the late 18th century.</answer>",
            "question": "Describe the Industrial Revolution",
            "answer": "The Industrial Revolution was a period of major industrialization that began in Britain in the late 18th century.",
            "task_type": "para_mixup",
        }
    ]
    
    print(f"\n=== Testing Multi-Task Batch Scoring ===")
    scores = compute_score(samples)
    for i, (sample, score) in enumerate(zip(samples, scores)):
        print(f"\n=== Sample {i+1} ({sample['task_type']}) ===")
        print(f"Response: {sample['response'][:80]}...")
        print(f"Scores: {score}")