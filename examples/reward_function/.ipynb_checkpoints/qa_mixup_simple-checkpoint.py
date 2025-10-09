# -*- coding: utf-8 -*-
import os
import re
import torch
from dataclasses import dataclass
from typing import Any, Dict
import regex

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

def extract_answer_content(response: str) -> str:
    """Extract content from <answer>...</answer> tags"""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

def length_based_quality_score(answer: str) -> float:
    """
    Simple length-based quality scoring for debugging
    Returns score between 0 and 1 based on answer length
    """
    if not answer or len(answer.strip()) == 0:
        return 0.0
    
    length = len(answer.strip())
    
    # Define length-based scoring rules
    if length >= 50:           # Long, detailed answers
        return 1.0
    elif length >= 20:         # Medium length answers  
        return 0.8
    elif length >= 10:         # Short but reasonable answers
        return 0.6
    elif length >= 5:          # Very short answers
        return 0.4
    elif length >= 1:          # Single character/word answers
        return 0.2
    else:                      # Empty answers
        return 0.0


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

# Global variables to store models (avoiding Ray Actor complexity for debugging)
_alignment_clf = None
_embedding_model = None
_weights = None
_initialized = False

def init_shared_reward_system(**kwargs):
    """Initialize the reward system in current process (no Ray Actor)"""
    global _alignment_clf, _embedding_model, _weights, _initialized
    
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
        
        # Load alignment classifier
        alignment_model_path = kwargs.get('alignment_model_path', "/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt")
        print(f"Loading alignment classifier from {alignment_model_path}...")
        _alignment_clf = get_alignment_classifier(alignment_model_path)
        print("✅ Alignment classifier loaded")
        
        # Load embedding model  
        embedding_model_path = kwargs.get('embedding_model_path', "/root/autodl-tmp/EasyR1/ckpt/bge-m3")
        print(f"Loading embedding model from {embedding_model_path}...")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(embedding_model_path)
        print("✅ Embedding model loaded")
        
        _initialized = True
        print("✅ Reward system initialization complete!")
        
    except Exception as e:
        print(f"❌ Reward system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def compute_score(reward_input: Dict[str, Any], format_weight: float = 0.1) -> Dict[str, float]:
    """
    Compute reward score directly in current process
    """
    global _alignment_clf, _embedding_model, _weights, _initialized
    
    if not _initialized:
        raise RuntimeError("Reward system not initialized. Call init_shared_reward_system() first.")
    
    # Normalize weights
    w_dict = {
        'quality': _weights['quality'],
        'alignment': _weights['alignment'], 
        'format': format_weight
    }
    total = w_dict['quality'] + w_dict['alignment'] + w_dict['format']
    w_normalized = {k: v/total for k, v in w_dict.items()}
    response = reward_input.get("response", "")
    answer_part = extract_answer_content(response)
    # 🚀 Quality score - using simple length-based scoring instead of LLM
    try:
        q_score = length_based_quality_score(answer_part)
        #print(f"🔍 Quality score for '{answer_part}...': {q_score} (length: {len(answer_part)})")
    except Exception as e:
        print(f"⚠️ Quality scoring failed: {e}")
        q_score = 0.5
    
    # Format score
    f_score = format_reward(response)
    if f_score<1:
        print(reward_input)
        print(f"格式分数：{f_score}")
    # Alignment score
    try:
        a_score = alignment_reward(answer_part, _alignment_clf, _embedding_model)
        #print(f"对齐分数：{a_score}")
    except Exception as e:
        print(f"⚠️ Alignment scoring failed: {e}")
        a_score = 0.5
    
    overall = w_normalized['quality'] * q_score + w_normalized['format'] * f_score + w_normalized['alignment'] * a_score
    return {
        "overall": float(overall),
        "quality": float(q_score),
        "format": float(f_score),
        "alignment": float(a_score),
    }

def alignment_reward(pair: str, alignment_clf, embedding_model) -> float:
    """Compute alignment reward using embedding + MLP classifier"""
    features = embedding_model.encode([pair], normalize_embeddings=True)[0]
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = alignment_clf(input_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = int(torch.argmax(probs).item())
    
    class_to_score = {0: 0.0, 1: 0.2, 2: 1.0}
    return class_to_score.get(pred_label, 0.0)

# Usage example:
if __name__ == "__main__":
    # Initialize once at the beginning of your training script
    init_shared_reward_system(
        alignment_model_path="/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt",
        embedding_model_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
        weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
    )
    
    # Test samples
    samples = [
        {
            "response": "<think>2+2 is basic arithmetic</think><answer>4</answer>",
            "question": "What is 2+2?",
            "answer": "4",
        },
        {
            "response": "<think>This is a complex mathematical problem that requires careful consideration</think><answer>The answer involves multiple steps of calculation</answer>",
            "question": "What is the derivative of x^2?",
            "answer": "2x",
        },
        {
            "response": "bad format without proper tags",
            "question": "Who wrote Hamlet?",
            "answer": "Shakespeare",
        }
    ]
    
    for i, sample in enumerate(samples):
        print(f"\n=== Sample {i+1} ===")
        score = compute_score(sample)
        print(f"Response: {sample['response'][:60]}...")
        print(f"Scores: {score}")