# -*- coding: utf-8 -*-
"""
qa_mixup.py - Fixed version with proper Ray actor handling
"""
import os
import re
import json
import torch
import ray
from dataclasses import dataclass
from typing import Any, Dict, Optional
import regex

JSON_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE
)

_default_quality_threshold = 4

def strip_tag_spaces(s: str) -> str:
    return re.sub(r"\s*(<|>|/)\s*", r"\1", s)

def compress_overall_10_to_5(score_10: int) -> int:
    if score_10 <= 4:
        return 0
    elif score_10 >= 9:
        return 5
    else:
        return score_10 - 4

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

def format_reward(response: str) -> float:
    return 1.0 if re.fullmatch(FORMAT_RE, response or "") else 0.0

def extract_answer_content(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

# Define the Ray actor at module level with a unique name
@ray.remote(num_gpus=1)
class QAMixupSharedRewardModel:
    """Ray Actor that holds the models in memory and serves all workers"""
    
    def __init__(
        self,
        model_path: str,
        alignment_model_path: str,
        embedding_model_path: str,
        weights: dict = None,
        use_bnb_4bit: bool = True,
        **kwargs
    ):
        print(f"🚀 Initializing QAMixupSharedRewardModel Actor on PID {os.getpid()}...")
        
        # Import required modules inside the actor
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from sentence_transformers import SentenceTransformer
        import torch.nn as nn
        import regex
        import json
        
        # Store regex patterns as instance variables
        self.JSON_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)
        self.FORMAT_RE = re.compile(
            r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
            re.DOTALL | re.IGNORECASE
        )
        
        # Initialize weights
        if weights is None:
            weights = {'quality': 0.6, 'alignment': 0.3, 'format': 0.1}
        
        total = weights['quality'] + weights['alignment'] + weights['format']
        self.weights = {
            'quality': weights['quality'] / total,
            'alignment': weights['alignment'] / total,
            'format': weights['format'] / total
        }
        
        # Define MLPClassifier inside the actor
        class MLPClassifier(nn.Module):
            def __init__(self, input_dim, hidden_dim=256):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 3)
                )
            def forward(self, x):
                return self.model(x)
        
        # Load alignment classifier
        print("Loading alignment classifier...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(alignment_model_path, map_location=device)
        input_dim = ckpt["input_dim"]
        state_dict = ckpt["model_state_dict"]
        self.alignment_clf = MLPClassifier(input_dim=input_dim).to(device)
        self.alignment_clf.load_state_dict(state_dict)
        self.alignment_clf.eval()
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        
        # Load LLM
        print("Loading LLM model...")
        self._init_llm(model_path, use_bnb_4bit, **kwargs)
        
        print("✅ QAMixupSharedRewardModel initialization complete!")
    
    def _init_llm(self, model_path, use_bnb_4bit, **kwargs):
        """Initialize LLM model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        self.max_new_tokens = kwargs.get('max_new_tokens', 128)
        self.temperature = kwargs.get('temperature', 1.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.top_k = kwargs.get('top_k', 50)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        self.length_penalty = kwargs.get('length_penalty', 1.0)
        self.retry_n = kwargs.get('retry_n', 5)
        
        device_map = kwargs.get('device_map', 'auto')
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        
        if use_bnb_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generate prompts
        self.system_prompt = (
            "As a data quality estimator, your task is to assess the quality of data sample "
            "based on the criteria: Rarity, Complexity, Informativeness. "
            "Please rate the sample on a scale from 1 to 10 for each criterion, and return an overall rating on a "
            "scale from 1 to 10, where a higher score indicates higher level of quality. "
            "Ensure that the ratings are not overly concentrated around a specific score. "
            "If multiple samples have similar qualities, consider spreading the scores more evenly to reflect subtle differences."
        )
        self.user_prompt = (
            "Please carefully evaluate the following data sample and return the integral evaluation scores using the JSON format:\n"
            "{\n"
            '  "Rarity": <number, 1-10>,\n'
            '  "Complexity": <number, 1-10>,\n'
            '  "Informativeness": <number, 1-10>,\n'
            '  "Overall": <number, 1-10>\n'
            "}\n"
        )
    
    def _build_llama31_prompt(self, system_prompt: str, user_prompt: str, conversation: str) -> str:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
            f"{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>"
            f"{user_prompt}\n## Data sample (conversation):\n{conversation}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        )
    
    def _extract_overall10_from_text(self, text: str) -> Optional[int]:
        import re
        import json
        
        m = self.JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                overall = obj.get("Overall", obj.get("Overall rating", None))
                if overall is not None:
                    val = int(overall)
                    if 1 <= val <= 10:
                        return val
            except Exception:
                pass

        m = re.search(r'"Overall"\s*:\s*(\d+)', text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                return val

        m = re.search(r'"Overall rating"\s*:\s*(\d+)', text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                return val

        return None
    
    def rate(self, answer_part: str) -> Dict[str, Any]:
        """Rate the quality of an answer"""
        import torch
        
        prompt = self._build_llama31_prompt(self.system_prompt, self.user_prompt, answer_part)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        out_text = None
        overall_10 = None

        for _ in range(self.retry_n):
            with torch.no_grad():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    length_penalty=self.length_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )
            gen_text = self.tokenizer.decode(
                gen[0][enc['input_ids'].shape[1]:], skip_special_tokens=True
            )
            out_text = gen_text

            overall_10 = self._extract_overall10_from_text(out_text)
            if overall_10 is not None:
                break

        if overall_10 is None:
            overall_10 = 5
        overall_5 = compress_overall_10_to_5(overall_10)

        return {
            "overall_10": overall_10,
            "overall_5": overall_5,
            "raw_text": out_text or "",
        }
    
    def compute_score(self, reward_input: Dict[str, Any], format_weight: float = 0.1) -> Dict[str, float]:
        """Compute reward score for a single sample"""
        import torch
        import re
        
        # Normalize weights
        total = self.weights['quality'] + self.weights['alignment'] + format_weight
        w = {
            'quality': self.weights['quality'] / total,
            'alignment': self.weights['alignment'] / total,
            'format': format_weight / total
        }
        
        response = reward_input.get("response", "")
        
        # Extract answer content
        match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if match:
            answer_part = match.group(1).strip()
        else:
            answer_part = response.strip()
        
        # Quality score
        try:
            rating_res = self.rate(answer_part)
            qual_5 = int(rating_res["overall_5"])
            
            if qual_5 >= _default_quality_threshold:
                q_score = 1.0
            elif qual_5 == 3:
                q_score = 0.2
            else:
                q_score = 0.0
        except Exception as e:
            print(f"⚠️ Quality scoring failed: {e}")
            import traceback
            traceback.print_exc()
            q_score = 0.5
        
        # Format score
        f_score = 1.0 if re.fullmatch(self.FORMAT_RE, response or "") else 0.0
        
        # Alignment score
        try:
            features = self.embedding_model.encode([answer_part], normalize_embeddings=True)[0]
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.alignment_clf.model[0].weight.device)
            
            with torch.no_grad():
                logits = self.alignment_clf(input_tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                pred_label = int(torch.argmax(probs).item())
            
            class_to_score = {0: 0.0, 1: 0.2, 2: 1.0}
            a_score = class_to_score.get(pred_label, 0.0)
        except Exception as e:
            print(f"⚠️ Alignment scoring failed: {e}")
            import traceback
            traceback.print_exc()
            a_score = 0.5
        
        overall = w['quality'] * q_score + w['format'] * f_score + w['alignment'] * a_score
        
        return {
            "overall": float(overall),
            "quality": float(q_score),
            "format": float(f_score),
            "alignment": float(a_score),
        }


# Global actor handle - store with a unique name to avoid conflicts
_qa_mixup_shared_model_actor = None
_actor_initialized = False

def init_reward_system(**kwargs):
    """Initialize the shared reward model actor with a unique namespace"""
    global _qa_mixup_shared_model_actor, _actor_initialized
    
    if not _actor_initialized:
        print(f"🚀 PID {os.getpid()}: Initializing QA Mixup shared reward system...")
        
        # Check if Ray is initialized
        if not ray.is_initialized():
            print("Ray not initialized in reward function, assuming it's initialized elsewhere")
        
        # Try to get existing actor or create new one
        actor_name = "qa_mixup_reward_model"
        try:
            # Try to get existing named actor
            _qa_mixup_shared_model_actor = ray.get_actor(actor_name)
            print(f"✅ Found existing QA Mixup reward model actor: {actor_name}")
        except ValueError:
            # Actor doesn't exist, create it
            print(f"Creating new QA Mixup reward model actor: {actor_name}")
            
            # Map parameters properly
            weights = kwargs.get('weights', RewardWeights())
            if hasattr(weights, '__dict__'):
                weights_dict = {'quality': weights.quality, 'alignment': weights.alignment, 'format': weights.format}
            else:
                weights_dict = weights
            
            actor_kwargs = {
                'model_path': kwargs.get('model_path', "/root/autodl-tmp/EasyR1/ckpt/Llama-3.1-8B-Instruct"),
                'alignment_model_path': kwargs.get('alignment_model_path', "/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt"),
                'embedding_model_path': kwargs.get('embedding_model_path', "/root/autodl-tmp/EasyR1/ckpt/bge-m3"),
                'weights': weights_dict,
                'use_bnb_4bit': kwargs.get('use_bnb_4bit', True),
            }
            
            # Add optional kwargs
            for key in ['device_map', 'torch_dtype', 'max_new_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'length_penalty', 'retry_n']:
                if key in kwargs:
                    actor_kwargs[key] = kwargs[key]
            
            # Create named actor
            _qa_mixup_shared_model_actor = QAMixupSharedRewardModel.options(
                name=actor_name,
                lifetime="detached",  # Keep actor alive even if creator dies
                max_restarts=3,
                max_task_retries=3,
            ).remote(**actor_kwargs)
            
            print(f"✅ QA Mixup shared reward model actor created: {actor_name}")
        
        _actor_initialized = True
    else:
        print(f"🔄 PID {os.getpid()}: QA Mixup shared reward system already initialized")
    
    return _qa_mixup_shared_model_actor


def compute_score(reward_input: Dict[str, Any], format_weight: float = 0.1) -> Dict[str, float]:
    """
    Compute reward score using the shared model actor.
    Automatically initializes the actor if needed.
    """
    global _qa_mixup_shared_model_actor, _actor_initialized
    
    # Auto-initialize if not done yet
    if not _actor_initialized or _qa_mixup_shared_model_actor is None:
        print("⚠️ Reward system not initialized, auto-initializing with default parameters...")
        init_reward_system(
            model_path="/root/autodl-tmp/EasyR1/ckpt/Llama-3.1-8B-Instruct",
            alignment_model_path="/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt",
            embedding_model_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
            weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
            use_bnb_4bit=True,
        )
    
    if _qa_mixup_shared_model_actor is None:
        raise RuntimeError("Failed to initialize shared reward model actor")
    
    # Use ray.get to call the remote actor method
    try:
        return ray.get(_qa_mixup_shared_model_actor.compute_score.remote(reward_input, format_weight))
    except Exception as e:
        print(f"❌ Error computing score: {e}")
        import traceback
        traceback.print_exc()
        # Return default scores on error
        return {
            "overall": 0.5,
            "quality": 0.5,
            "format": 0.5,
            "alignment": 0.5,
        }


# Module initialization - call init when module is imported
# This ensures the actor is ready when compute_score is called
if __name__ != "__main__":
    # Only auto-init when imported as a module, not when run as script
    pass  # Don't auto-init, let the training script handle it


if __name__ == "__main__":
    # Test code
    import torch
    
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init()
    
    # Initialize the reward system
    init_reward_system(
        model_path="/root/autodl-tmp/EasyR1/ckpt/Llama-3.1-8B-Instruct",
        alignment_model_path="/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt",
        embedding_model_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
        weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
        use_bnb_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    # Test sample
    sample = {
        "response": "<think>Let me think about this</think><answer>The answer is 4</answer>",
        "question": "What is 2+2?",
        "answer": "4",
    }
    
    score = compute_score(sample)
    print(f"Score: {score}")
    
    # Clean up
    ray.shutdown()