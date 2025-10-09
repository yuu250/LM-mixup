# -*- coding: utf-8 -*-
import os
import re
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import regex

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

JSON_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE
)
ANSWER_RE = re.compile(
    r"<answer>.*?Q:.*?A:.*?</answer>",
    re.DOTALL | re.IGNORECASE
)

def extract_answer_content(response: str) -> str:
    """Extract content from <answer>...</answer> tags"""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

def compress_overall(vals):
    """Compress score from 1-10 to 0-5 scale"""
    comp = min(9, max(4, vals)) - 4
    return comp

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
    # Use current CUDA device if available (respects Ray's CUDA_VISIBLE_DEVICES)
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device = f"cuda:{device}"
    else:
        device = "cpu"
    
    print(f"🔧 Loading alignment classifier to device: {device}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if "input_dim" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint must contain 'input_dim' and 'model_state_dict'.")

    input_dim = ckpt["input_dim"]
    state_dict = ckpt["model_state_dict"]

    model = MLPClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ Alignment classifier loaded on device: {device}")
    return model

def alignment_reward(pair: str, alignment_clf, embedding_model) -> float:
    """Compute alignment reward using embedding + MLP classifier"""
    features = embedding_model.encode([pair], normalize_embeddings=True)[0]
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    # Ensure tensor is on the same device as the model
    device = next(alignment_clf.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        logits = alignment_clf(input_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = int(torch.argmax(probs).item())
    
    class_to_score = {0: 0.0, 1: 0.2, 2: 1.0}
    return class_to_score.get(pred_label, 0.0)

def extract_overall_score_from_local_response(response: str) -> Optional[int]:
    """Extract Overall score from local model response JSON"""
    # Try to find JSON in the response
    json_match = JSON_RE.search(response)
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
    overall_pattern = re.search(r'"Overall"\s*:\s*(\d+)', response, re.IGNORECASE)
    if overall_pattern:
        try:
            score = int(overall_pattern.group(1))
            if 1 <= score <= 10:
                return score
        except ValueError:
            pass
    
    return None

def local_quality_score(answer: str, local_model, tokenizer, generation_kwargs: Dict[str, Any]) -> float:
    """
    Use local LLM to evaluate answer quality
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
    
    # Format prompt for chat model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Tokenize and generate
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(local_model.device)
        
        # Generate response
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": generation_kwargs.get("max_new_tokens", 512),
                "temperature": generation_kwargs.get("temperature", 0.1),
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            outputs = local_model.generate(**inputs, **gen_kwargs)
            
        # Decode response
        response_ids = outputs[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Extract score
        overall_score = extract_overall_score_from_local_response(response)
        if overall_score is not None:
            overall_score = compress_overall(overall_score)
            if overall_score >= 4:      # High quality: 8-10 -> 1.0
                return 1.0
            elif overall_score == 3:    # Medium-high: 6-7 -> 0.3
                return 0.3
            else:                       # Low: 1-5 -> 0.0
                return 0.0
        else:
            print(f"⚠️ Failed to extract score from local model response: {response[:100]}...")
            return 0.0
            
    except Exception as e:
        print(f"⚠️ Local model call failed: {e}")
        return 0.0

# Global variables to store models
_alignment_clf = None
_embedding_model = None
_local_model = None
_local_tokenizer = None
_generation_kwargs = None
_weights = None
_initialized = False

def init_shared_reward_system(**kwargs):
    """Initialize the reward system with local model support and Ray GPU management"""
    global _alignment_clf, _embedding_model, _local_model, _local_tokenizer, _generation_kwargs, _weights, _initialized
    
    if _initialized:
        print(f"🔄 PID {os.getpid()}: Reward system already initialized, skipping")
        return
        
    # Get Ray worker GPU allocation info
    import ray
    if ray.is_initialized():
        try:
            ray_resources = ray.available_resources()
            current_gpu = None
            # Ray automatically assigns CUDA_VISIBLE_DEVICES for GPU workers
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                current_gpu = os.environ['CUDA_VISIBLE_DEVICES']
                print(f"🔧 Ray assigned GPU: {current_gpu}")
            print(f"📊 Available Ray resources: {ray_resources}")
        except Exception as e:
            print(f"⚠️ Could not get Ray resource info: {e}")
        
    print(f"🚀 PID {os.getpid()}: Initializing local reward system...")
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
        
        # Load local reward model if specified
        use_local_model = kwargs.get('use_local_reward_model', False)
        if use_local_model:
            local_model_path = kwargs.get('local_reward_model_path')
            if local_model_path:
                print(f"Loading local reward model from {local_model_path}...")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                _local_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                if _local_tokenizer.pad_token is None:
                    _local_tokenizer.pad_token = _local_tokenizer.eos_token
                
                model_kwargs = kwargs.get('local_reward_model_kwargs', {}).copy()
                
                # Separate model loading args from generation args
                generation_only_keys = ['max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 
                                      'num_beams', 'early_stopping', 'pad_token_id', 'eos_token_id']
                generation_kwargs = {}
                for key in generation_only_keys:
                    if key in model_kwargs:
                        generation_kwargs[key] = model_kwargs.pop(key)
                
                # Store generation kwargs for later use
                global _generation_kwargs
                _generation_kwargs = generation_kwargs
                
                # Ensure proper device placement for Ray-managed GPUs
                if torch.cuda.is_available() and 'device_map' not in model_kwargs:
                    model_kwargs['device_map'] = 'auto'
                
                # Handle quantization configuration
                use_quantization = kwargs.get('use_quantization', False)
                if use_quantization:
                    try:
                        from transformers import BitsAndBytesConfig
                        
                        quant_config = kwargs.get('quantization_config', {})
                        # Convert string dtype to torch dtype
                        if 'bnb_4bit_compute_dtype' in quant_config:
                            dtype_str = quant_config['bnb_4bit_compute_dtype']
                            if isinstance(dtype_str, str):
                                if dtype_str == 'bfloat16':
                                    quant_config['bnb_4bit_compute_dtype'] = torch.bfloat16
                                elif dtype_str == 'float16':
                                    quant_config['bnb_4bit_compute_dtype'] = torch.float16
                                elif dtype_str == 'float32':
                                    quant_config['bnb_4bit_compute_dtype'] = torch.float32
                        
                        bnb_config = BitsAndBytesConfig(**quant_config)
                        model_kwargs['quantization_config'] = bnb_config
                        print(f"🔧 Using quantization config: {quant_config}")
                        
                    except ImportError:
                        print("⚠️ BitsAndBytesConfig not available, skipping quantization")
                        use_quantization = False
                
                _local_model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    **model_kwargs
                )
                
                # Print actual device allocation and quantization status
                if hasattr(_local_model, 'device'):
                    print(f"✅ Local reward model loaded on device: {_local_model.device}")
                else:
                    print("✅ Local reward model loaded with device_map=auto")
                    
                if use_quantization:
                    print("✅ Model loaded with 4-bit quantization")
                else:
                    print("✅ Model loaded without quantization")
            else:
                print("⚠️ use_local_reward_model is True but local_reward_model_path not provided")
        
        # Load alignment classifier
        alignment_model_path = kwargs.get('alignment_model_path', "/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt")
        if os.path.exists(alignment_model_path):
            print(f"Loading alignment classifier from {alignment_model_path}...")
            _alignment_clf = get_alignment_classifier(alignment_model_path)
            print("✅ Alignment classifier loaded")
        else:
            print(f"⚠️ Alignment model not found at {alignment_model_path}, skipping")
        
        # Load embedding model  
        embedding_model_path = kwargs.get('embedding_model_path', "/root/autodl-tmp/EasyR1/ckpt/bge-m3")
        if os.path.exists(embedding_model_path):
            print(f"Loading embedding model from {embedding_model_path}...")
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(embedding_model_path)
            print("✅ Embedding model loaded")
        else:
            print(f"⚠️ Embedding model not found at {embedding_model_path}, skipping")
        
        _initialized = True
        print("✅ Local reward system initialization complete!")
        
    except Exception as e:
        print(f"❌ Reward system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def compute_score(reward_input: Dict[str, Any], format_weight: float = 0.1, **kwargs) -> Dict[str, float]:
    """
    Compute reward score using local model for quality assessment
    """
    global _alignment_clf, _embedding_model, _local_model, _local_tokenizer, _generation_kwargs, _weights, _initialized
    
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
    
    # Quality score - using local model if available
    try:
        if _local_model is not None and _local_tokenizer is not None:
            gen_kwargs = _generation_kwargs or {}
            q_score = local_quality_score(answer_part, _local_model, _local_tokenizer, gen_kwargs)
            print(f"🔍 Local model Quality score for '{answer_part[:50]}...': {q_score}")
        else:
            print("⚠️ Local model not available, using length-based fallback")
            # Fallback to length-based scoring
            q_score = min(1.0, len(answer_part) / 100.0)
    except Exception as e:
        print(f"⚠️ Local model quality scoring failed: {e}")
        q_score = 0.5
    
    # Format score
    f_score = format_reward(response)
    if f_score < 1:
        print(f"格式分数：{f_score}")
    
    # Alignment score
    try:
        if _alignment_clf is not None and _embedding_model is not None:
            a_score = alignment_reward(answer_part, _alignment_clf, _embedding_model)
        else:
            print("⚠️ Alignment models not available, using default score")
            a_score = 0.5
    except Exception as e:
        print(f"⚠️ Alignment scoring failed: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
        # Fallback to simple scoring
        a_score = 0.5
    
    overall = w_normalized['quality'] * q_score + w_normalized['format'] * f_score + w_normalized['alignment'] * a_score
    
    return {
        "overall": float(overall),
        "quality": float(q_score),
        "format": float(f_score),
        "alignment": float(a_score),
    }

# Usage example:
if __name__ == "__main__":
    # Initialize with local model
    init_shared_reward_system(
        use_local_reward_model=True,
        local_reward_model_path="/path/to/your/local/model",
        local_reward_model_kwargs={"device_map": "auto", "torch_dtype": "float16"},
        alignment_model_path="/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt",
        embedding_model_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
        weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
    )
    
    # Test sample
    sample = {
        "response": "<think>This requires merging related Q&A about machine learning</think><answer>Q: What is machine learning and how does it work? A: Machine learning is a subset of artificial intelligence that uses algorithms to automatically learn patterns from data without explicit programming.</answer>",
        "response_length": 200,
        "ground_truth": "Standard ML answer",
    }
    
    print(f"\n=== Test Sample ===")
    score = compute_score(sample)
    print(f"Response: {sample['response'][:60]}...")
    print(f"Scores: {score}")