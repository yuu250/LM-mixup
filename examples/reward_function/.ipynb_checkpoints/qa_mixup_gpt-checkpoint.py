# -*- coding: utf-8 -*-
import os
import re
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict
import regex
import requests
import time
def parse_openai_resp(resp: dict) -> str:
    """安全地提取 assistant 返回文本；若失败抛 ValueError。"""
    if "choices" not in resp:
        raise ValueError(f"No 'choices' field: {resp}")

    choice0 = resp["choices"][0]

    # Chat 流程的普通文本回复
    if "message" in choice0 and "content" in choice0["message"] \
            and choice0["message"]["content"] is not None:
        return choice0["message"]["content"]

    # text-completion 旧格式
    if "text" in choice0:
        return choice0["text"]

    # Function / tool call 等
    if "message" in choice0 and ("tool_calls" in choice0["message"]
                                 or "function_call" in choice0["message"]):
        raise ValueError(f"Model returned a tool/function call: {resp}")

    # 兜底：未知格式
    raise ValueError(f"Unrecognized response schema: {resp}")

# ---------------- 2. GPT 调用 ----------------
def gpt_call(prompt: str, *, api_key: str = "sk-**",
            url: str = "https://api2.aigcbest.top/v1/chat/completions",
             model: str = "gpt-4o-mini",
             retries: int = 8, delay: int = 25) -> str:

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # 强行要求返回纯文本，避免 tool call
        "response_format": {"type": "text"}
    }

    headers = {
        "Accept": "application/json",
        "Authorization": api_key,
        "Content-Type": "application/json"
    }

    for i in range(1, retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=25)
            r.raise_for_status()
            return parse_openai_resp(r.json())

        except (ValueError,                       
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            print(f"[Retry {i}/{retries}] {e}")
            time.sleep(delay)

    return "exception"

    
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
    
def compress_overall(vals):
    #print("Original distribution :", Counter(vals))
    comp = min(9, max(4, vals)) - 4
    #print("Compressed distribution:", Counter(comp))
    return comp

def gpt4_quality_score(answer: str) -> float:
    """
    Use GPT-4 API to evaluate answer quality
    Returns normalized score between 0 and 1
    """
    if not answer or len(answer.strip()) == 0:
        return 0.0
    
    # Construct the evaluation prompt
    system_prompt = ("As a data quality estimator, your task is to assess the quality of data sample "
                    "based on the criteria: Rarity, Complexity, Informativeness. "
                    "Please rate the sample on a scale from 1 to 10 for each criterion, and return an overall rating on a "
                    "scale from 1 to 10, where a higher score indicates higher level of quality. "
                    "Ensure that the ratings are not overly concentrated around a specific score. "
                    "If multiple samples have similar qualities, consider spreading the scores more evenly to reflect subtle differences."
                    "IMPORTANT: If the input is not a valid QA pair (i.e., it does not include both a meaningful question and a meaningful answer), "
                    "then assign a score of 0 to all four categories."
                    )
    
    user_prompt = ("Please carefully evaluate the following QA and return the integral evaluation scores using the JSON format:\n"
                  "{\n"
                  '  "Rarity": <number, 1-10>,\n'
                  '  "Complexity": <number, 1-10>,\n'
                  '  "Informativeness": <number, 1-10>,\n'
                  '  "Overall": <number, 1-10>\n'
                  "}\n\n"
                  f"Data sample to evaluate:\n{answer}")
    
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        # Call GPT-4 API
        gpt_response = gpt_call(prompt=full_prompt,api_key="sk-hoUItnzIvpOkw23I0FfONQJqRCSLlX6OcoSHuWJ4r4AHnwAW")
        
        # Extract JSON from response
        overall_score = extract_overall_score_from_gpt_response(gpt_response)
        overall_score = compress_overall(overall_score)
        if overall_score is not None:
            if overall_score >= 4:      # High quality: 8-10 -> 1.0
                return 1.0
            elif overall_score == 3:    # Medium-high: 6-7 -> 0.8
                return 0.3
            else:                       # Low: 1 -> 0.2
                return 0.0
        else:
            print(f"⚠️ Failed to extract score from GPT response: {gpt_response[:100]}...")
            return 0.0
            
    except Exception as e:
        print(f"⚠️ GPT-4 API call failed: {e}")
        return 0.0  

def extract_overall_score_from_gpt_response(gpt_response: str) -> int:
    """Extract Overall score from GPT response JSON"""
    # Try to find JSON in the response
    json_match = JSON_RE.search(gpt_response)
    if json_match:
        try:
            json_obj = json.loads(json_match.group(0))
            overall = json_obj.get("Overall", json_obj.get("overall"))
            if overall is not None:
                score = int(overall)
                if 1 <= score <= 10:
                    return score
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    # Fallback: try to find "Overall": number pattern
    overall_pattern = re.search(r'"Overall"\s*:\s*(\d+)', gpt_response, re.IGNORECASE)
    if overall_pattern:
        try:
            score = int(overall_pattern.group(1))
            if 1 <= score <= 10:
                return score
        except ValueError:
            pass
    
    return None

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

# Global variables to store models
_alignment_clf = None
_embedding_model = None
_weights = None
_initialized = False

def init_shared_reward_system(**kwargs):
    """Initialize the reward system in current process"""
    global _alignment_clf, _embedding_model, _weights, _initialized
    
    if _initialized:
        print(f"🔄 PID {os.getpid()}: Reward system already initialized, skipping")
        return
        
    print(f"🚀 PID {os.getpid()}: Initializing GPT-4 reward system...")
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
        print("✅ GPT-4 reward system initialization complete!")
        
    except Exception as e:
        print(f"❌ Reward system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def compute_score(reward_input: Dict[str, Any], format_weight: float = 0.1) -> Dict[str, float]:
    """
    Compute reward score using GPT-4 for quality assessment
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
    
    # 🚀 Quality score - using GPT-4 API evaluation
    try:
        q_score = gpt4_quality_score(answer_part)
        print(f"🔍 GPT-4 Quality score for '{answer_part[:50]}...': {q_score}")
    except Exception as e:
        print(f"⚠️ GPT-4 quality scoring failed: {e}")
        q_score = 0.5
    
    # Format score
    f_score = format_reward(response)
    if f_score < 1:
        print(f"格式分数：{f_score}")
    
    # Alignment score
    try:
        a_score = alignment_reward(answer_part, _alignment_clf, _embedding_model)
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
    # Mock GPT-4 call function for testing
    def mock_gpt_4_call(prompt):
        # This is just a mock for testing - replace with your actual GPT-4 API call
        return '''
        {
            "Rarity": 7,
            "Complexity": 8,
            "Informativeness": 6,
            "Overall": 7
        }
        '''
    
    # Initialize with GPT-4 function
    init_shared_reward_system(
        gpt_4_call=mock_gpt_4_call,
        alignment_model_path="/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt",
        embedding_model_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
        weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
    )
    
    # Test samples
    samples = [
        {
            "response": "<think>This requires merging related Q&A about machine learning</think><answer>Q: What is machine learning and how does it work? A: Machine learning is a subset of artificial intelligence that uses algorithms to automatically learn patterns from data without explicit programming.</answer>",
            "question": "Merge ML questions",
            "answer": "Merged answer",
        }
    ]
    
    for i, sample in enumerate(samples):
        print(f"\n=== Sample {i+1} ===")
        score = compute_score(sample)
        print(f"Response: {sample['response'][:60]}...")
        print(f"Scores: {score}")