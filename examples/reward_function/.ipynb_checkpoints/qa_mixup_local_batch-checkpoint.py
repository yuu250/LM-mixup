# -*- coding: utf-8 -*-
import os
import re
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List
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

def batch_alignment_reward(pairs: List[str], alignment_clf, embedding_model) -> List[float]:
    """Compute alignment rewards for multiple pairs using embedding + MLP classifier"""
    if not pairs:
        return []
    
    # Get embeddings for all pairs at once
    features = embedding_model.encode(pairs, normalize_embeddings=True)
    input_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Ensure tensor is on the same device as the model
    device = next(alignment_clf.parameters()).device
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        logits = alignment_clf(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1)
    
    class_to_score = {0: 0.0, 1: 0.2, 2: 1.0}
    scores = [class_to_score.get(int(label.item()), 0.0) for label in pred_labels]
    return scores

def extract_overall_score_from_local_response(response: str) -> Optional[int]:
    """
    Extract Overall score from local model response with multiple fallback strategies
    Handles various formats the model might output
    """
    response = response.strip()
    
    # Strategy 1: Try to find JSON in the response
    json_match = JSON_RE.search(response)
    if json_match:
        try:
            json_obj = json.loads(json_match.group(0))
            overall = json_obj.get("Overall", json_obj.get("overall", json_obj.get("OVERALL")))
            if overall is not None:
                score = int(overall)
                if 1 <= score <= 10:
                    return score
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    
    # Strategy 2: Find "Overall": number pattern (case insensitive, multiple variants)
    overall_patterns = [
        # JSON-like patterns
        r'"Overall"\s*:\s*(\d+)',
        r'"overall"\s*:\s*(\d+)', 
        r'"OVERALL"\s*:\s*(\d+)',
        # Plain text patterns  
        r'Overall\s*:\s*(\d+)',
        r'overall\s*:\s*(\d+)',
        r'OVERALL\s*:\s*(\d+)',
        # With additional words
        r'Overall\s+Score\s*:\s*(\d+)',
        r'overall\s+score\s*:\s*(\d+)',
        r'Overall\s+Rating\s*:\s*(\d+)', 
        r'overall\s+rating\s*:\s*(\d+)',
        r'Overall\s+Quality\s*:\s*(\d+)',
        r'overall\s+quality\s*:\s*(\d+)',
        # With parentheses like "Overall (O):"
        r'Overall\s*\([^)]*\)\s*:\s*(\d+)',
        r'overall\s*\([^)]*\)\s*:\s*(\d+)',
    ]
    
    for pattern in overall_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                if 1 <= score <= 10:
                    return score
            except ValueError:
                continue
    
    # Strategy 3: Extract floating point scores like "8.5/10" or "Overall: 8.9/10" 
    float_patterns = [
        # /10 patterns
        r'Overall[:\s]*(\d+\.?\d*)/10',
        r'overall[:\s]*(\d+\.?\d*)/10',
        r'Overall\s+Score[:\s]*(\d+\.?\d*)/10',
        r'Overall\s+Rating[:\s]*(\d+\.?\d*)/10',
        r'Overall\s+Quality[:\s]*(\d+\.?\d*)/10',
        r'Overall\s*\([^)]*\)[:\s]*(\d+\.?\d*)/10',
        r'(\d+\.?\d*)/10\s*$',
        # End-of-line patterns
        r'Overall[:\s]*(\d+\.?\d*)\s*$',
        r'Overall\s+Score[:\s]*(\d+\.?\d*)\s*$',
        r'Overall\s+Rating[:\s]*(\d+\.?\d*)\s*$',
        r'Overall\s+Quality[:\s]*(\d+\.?\d*)\s*$',
        r'Overall\s*\([^)]*\)[:\s]*(\d+\.?\d*)\s*$',
    ]
    
    for pattern in float_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                score = float(match.group(1))
                if 1 <= score <= 10:
                    return int(round(score))  # Round to nearest integer
            except ValueError:
                continue
    
    # Strategy 4: Find any number between 1-10 at the end of response
    end_number = re.search(r'(\d+)\s*$', response.strip())
    if end_number:
        try:
            score = int(end_number.group(1))
            if 1 <= score <= 10:
                return score
        except ValueError:
            pass
    
    # Strategy 5: Find any standalone number between 1-10 in the response
    all_numbers = re.findall(r'\b(\d+)\b', response)
    for num_str in reversed(all_numbers):  # Check from end to beginning
        try:
            score = int(num_str)
            if 1 <= score <= 10:
                return score
        except ValueError:
            continue
    
    return None

def extract_batch_scores_from_local_response(response: str, expected_count: int) -> List[Optional[int]]:
    """Extract Overall scores from batch local model response"""
    scores = []
    
    # Try to find JSON array in the response
    json_matches = JSON_RE.findall(response)
    
    if json_matches:
        for match in json_matches:
            try:
                json_obj = json.loads(match)
                if isinstance(json_obj, list):
                    # Handle array format
                    for item in json_obj:
                        if isinstance(item, dict):
                            overall = item.get("Overall", item.get("overall"))
                            if overall is not None:
                                score = int(overall)
                                if 1 <= score <= 10:
                                    scores.append(score)
                                else:
                                    scores.append(None)
                            else:
                                scores.append(None)
                    break
                else:
                    # Handle single object
                    overall = json_obj.get("Overall", json_obj.get("overall"))
                    if overall is not None:
                        score = int(overall)
                        if 1 <= score <= 10:
                            scores.append(score)
                        else:
                            scores.append(None)
                    else:
                        scores.append(None)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    
    # Fallback: try to find multiple "Overall": number patterns
    if not scores:
        overall_patterns = re.findall(r'"Overall"\s*:\s*(\d+)', response, re.IGNORECASE)
        for match in overall_patterns:
            try:
                score = int(match)
                if 1 <= score <= 10:
                    scores.append(score)
                else:
                    scores.append(None)
            except ValueError:
                scores.append(None)
    
    # Pad or truncate to expected count
    while len(scores) < expected_count:
        scores.append(None)
    return scores[:expected_count]

def estimate_token_count(text: str, tokenizer=None) -> int:
    """Estimate token count for a text string"""
    if tokenizer:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except:
            pass
    # Fallback: rough estimation (1 token ≈ 4 characters for most languages)
    return len(text) // 4

def batch_local_quality_score(answers: List[str], local_model, tokenizer, generation_kwargs: Dict[str, Any]) -> List[float]:
    """
    Use local LLM to evaluate multiple answer qualities using true batch inference
    Each sample gets its own prompt but processed in batches for efficiency
    Returns list of normalized scores between 0 and 1
    """
    if not answers:
        return []
    
    # Filter out empty answers
    valid_indices = []
    valid_answers = []
    for i, answer in enumerate(answers):
        if answer and len(answer.strip()) > 0:
            valid_indices.append(i)
            valid_answers.append(answer)
    
    if not valid_answers:
        return [0.0] * len(answers)
    
    print(f"🚀 Starting true batch inference for {len(valid_answers)} samples")
    
    # Get batch size from config (default to 4 for small models)
    batch_size = generation_kwargs.get("batch_size", 4)
    print(f"📦 Using batch size: {batch_size}")
    
    # Process in batches
    all_scores = []
    for batch_start in range(0, len(valid_answers), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_answers))
        batch_answers = valid_answers[batch_start:batch_end]
        
        try:
            batch_scores = process_batch_inference(batch_answers, local_model, tokenizer, generation_kwargs)
            all_scores.extend(batch_scores)
            print(f"✅ Processed batch {batch_start//batch_size + 1}: {len(batch_answers)} samples")
        except Exception as e:
            print(f"⚠️ Batch {batch_start//batch_size + 1} failed: {e}, using fallback")
            # Fallback to individual processing for this batch
            fallback_scores = []
            for answer in batch_answers:
                try:
                    score = simple_local_quality_score(answer, local_model, tokenizer, generation_kwargs)
                    if score >= 1.0:
                        fallback_scores.append(9)
                    elif score >= 0.3:
                        fallback_scores.append(7)
                    else:
                        fallback_scores.append(3)
                except:
                    fallback_scores.append(5)  # Default score
            all_scores.extend(fallback_scores)
    
    # Initialize result array with zeros
    result_scores = [0.0] * len(answers)
    
    # Fill in scores for valid answers
    for i, valid_idx in enumerate(valid_indices):
        if i < len(all_scores) and all_scores[i] is not None:
            compressed_score = compress_overall(all_scores[i])
            if compressed_score >= 4:      # High quality: 8-10 -> 1.0
                result_scores[valid_idx] = 1.0
            elif compressed_score == 3:    # Medium-high: 6-7 -> 0.3
                result_scores[valid_idx] = 0.3
            else:                          # Low: 1-5 -> 0.0
                result_scores[valid_idx] = 0.0
        else:
            result_scores[valid_idx] = 0.0
            
    return result_scores

def process_batch_inference(batch_answers: List[str], local_model, tokenizer, generation_kwargs: Dict[str, Any]) -> List[int]:
    """
    Process a batch of answers using true batch inference with correct left-padding
    Each answer gets its own prompt, but all are processed together in one forward pass
    """
    system_prompt = ("You are a data quality evaluator. Your task is to rate QA samples on a scale of 1-10. "
                    "CRITICAL: You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no text before or after. "
                    "Format: {\"Overall\": X} where X is an integer from 1 to 10. "
                    "Rate based on: Rarity (uniqueness), Complexity (depth), Informativeness (value). "
                    "Examples: "
                    "- Simple facts: {\"Overall\": 3} "
                    "- Good explanations: {\"Overall\": 7} "
                    "- Excellent analysis: {\"Overall\": 9}")
    
    # Prepare all prompts for the batch
    batch_messages = []
    for answer in batch_answers:
        user_prompt = f"Evaluate this QA sample:\n{answer}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        batch_messages.append(messages)
    
    # Convert to text format for tokenization
    batch_texts = []
    for messages in batch_messages:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_texts.append(input_text)
    
    try:
        # 🔧 Critical fix: Set left padding for GPT-style models
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        
        # 🔧 Critical fix: Use eos_token as pad_token to avoid random embeddings
        if tokenizer.pad_token is None or tokenizer.pad_token != tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"🔧 Set pad_token to eos_token: '{tokenizer.eos_token}'")
        
        # Tokenize all prompts with LEFT padding for batch processing
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True,  # LEFT padding now (set above)
            truncation=True,  # Truncate if too long
            max_length=4096  # Conservative max length for small models
        ).to(local_model.device)
        
        print(f"📊 Batch input shape: {inputs['input_ids'].shape}")
        print(f"📊 Padding side: {tokenizer.padding_side}")
        print(f"📊 Using pad_token_id: {tokenizer.pad_token_id} (eos_token_id: {tokenizer.eos_token_id})")
        
        # Generate responses for the entire batch
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": min(256, generation_kwargs.get("max_new_tokens", 256)),  # Conservative for small models
                "temperature": generation_kwargs.get("temperature", 0.1),
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,  # Use the same pad_token_id consistently
                # 🔧 Remove attention_mask from gen_kwargs since it's already in inputs
            }
            
            # True batch generation - all samples processed in parallel
            # inputs already contains input_ids, attention_mask, etc.
            outputs = local_model.generate(**inputs, **gen_kwargs)
            
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        
        # Decode each response in the batch
        batch_scores = []
        for i in range(len(batch_answers)):
            # Extract generated tokens for this sample
            input_length = inputs["input_ids"][i].shape[0]
            response_ids = outputs[i][input_length:]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Extract score from response
            score = extract_overall_score_from_local_response(response)
            
            # 🔧 Debug: Print raw LLM output when score extraction fails
            if score is None or score <= 0:
                print(f"⚠️ Sample {i+1} - Failed to extract score or got 0:")
                print(f"   Raw LLM response: '{response}'")
                print(f"   Input sample length: {len(batch_answers[i])}")
                print(f"   Input preview: {batch_answers[i][:100]}...")
                score = 5  # Default fallback
            
            batch_scores.append(score)
            
        print(f"✅ Batch inference completed with left-padding")
        return batch_scores
        
    except Exception as e:
        # Restore padding side on error
        if 'original_padding_side' in locals():
            tokenizer.padding_side = original_padding_side
        print(f"⚠️ Batch inference failed: {e}")
        raise

def process_chunk(chunk_answers: List[str], local_model, tokenizer, generation_kwargs: Dict[str, Any], system_prompt: str) -> List[int]:
    """Process a single chunk of answers and return their scores"""
    # Build sample texts for current chunk
    sample_texts = []
    for i, answer in enumerate(chunk_answers):
        sample_texts.append(f"Sample {i+1}:\n{answer}")
    
    user_prompt = ("Please carefully evaluate the following QA samples and return the integral evaluation scores using the JSON format:\n"
                  "[\n"
                  + ",\n".join([f'  {{"Rarity": <number, 1-10>, "Complexity": <number, 1-10>, "Informativeness": <number, 1-10>, "Overall": <number, 1-10>}}' for _ in chunk_answers]) + "\n"
                  "]\n\n"
                  "Data samples to evaluate:\n" + "\n\n".join(sample_texts))
    
    # Format prompt for chat model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        # Tokenize and generate
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(local_model.device)
        
        # Check if input is still too long
        if inputs["input_ids"].shape[1] > 100000:  # Conservative check
            raise RuntimeError(f"Input too long: {inputs['input_ids'].shape[1]} tokens")
        
        # Generate response
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": generation_kwargs.get("max_new_tokens", 1024),
                "temperature": generation_kwargs.get("temperature", 0.1),
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            outputs = local_model.generate(**inputs, **gen_kwargs)
            
        # Decode response
        response_ids = outputs[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Extract scores from batch response
        scores = extract_batch_scores_from_local_response(response, len(chunk_answers))
        return scores
        
    except Exception as e:
        print(f"⚠️ Chunk local model call failed: {e}")
        # Fallback to sequential processing for this chunk
        print("🔄 Falling back to sequential processing for this chunk...")
        try:
            # Use absolute import and inline implementation to avoid relative import issues
            chunk_scores = []
            for answer in chunk_answers:
                score = simple_local_quality_score(answer, local_model, tokenizer, generation_kwargs)
                # Convert to raw score (reverse the compression)
                if score >= 1.0:
                    chunk_scores.append(9)  # High quality
                elif score >= 0.3:
                    chunk_scores.append(7)  # Medium-high
                else:
                    chunk_scores.append(3)  # Low
            return chunk_scores
        except Exception as e2:
            print(f"⚠️ Sequential fallback also failed: {e2}")
            return [5] * len(chunk_answers)  # Default medium scores

def simple_local_quality_score(answer: str, local_model, tokenizer, generation_kwargs: Dict[str, Any]) -> float:
    """Simple single-answer quality scoring (inline to avoid import issues)"""
    if not answer or len(answer.strip()) == 0:
        return 0.0
    
    system_prompt = ("You are a data quality evaluator. Your task is to rate QA samples on a scale of 1-10. "
                    "CRITICAL: You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no text before or after. "
                    "Format: {\"Overall\": X} where X is an integer from 1 to 10. "
                    "Rate based on: Rarity (uniqueness), Complexity (depth), Informativeness (value). "
                    "Examples: "
                    "- Simple facts: {\"Overall\": 3} "
                    "- Good explanations: {\"Overall\": 7} "
                    "- Excellent analysis: {\"Overall\": 9}")
    
    user_prompt = f"Evaluate this QA sample:\n{answer}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(local_model.device)
        
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": min(512, generation_kwargs.get("max_new_tokens", 512)),
                "temperature": generation_kwargs.get("temperature", 0.1),
                "do_sample": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            outputs = local_model.generate(**inputs, **gen_kwargs)
            
        response_ids = outputs[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Extract score
        score = extract_overall_score_from_local_response(response)
        if score is not None:
            compressed_score = compress_overall(score)
            if compressed_score >= 4:
                return 1.0
            elif compressed_score == 3:
                return 0.3
            else:
                return 0.0
        else:
            # 🔧 Debug: Print raw LLM output when score extraction fails  
            print(f"⚠️ Simple local scoring - Failed to extract score:")
            print(f"   Raw LLM response: '{response}'")
            print(f"   Input sample: {answer[:100]}...")
            return 0.5  # Default for parsing failures
            
    except Exception as e:
        print(f"⚠️ Simple local scoring failed: {e}")
        return 0.5  # Default score

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
        
    print(f"🚀 PID {os.getpid()}: Initializing batch local reward system...")
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
                
                # 🔧 Critical fix: Always use eos_token as pad_token to avoid random embeddings
                _local_tokenizer.pad_token = _local_tokenizer.eos_token
                print(f"🔧 Set tokenizer pad_token to eos_token: '{_local_tokenizer.eos_token}'")
                
                model_kwargs = kwargs.get('local_reward_model_kwargs', {}).copy()
                
                # Separate model loading args from generation args
                generation_only_keys = ['max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample', 
                                      'num_beams', 'early_stopping', 'pad_token_id', 'eos_token_id', 'batch_size']
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
        print("✅ Batch local reward system initialization complete!")
        
    except Exception as e:
        print(f"❌ Reward system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1, **kwargs) -> List[Dict[str, float]]:
    """
    Compute reward scores for a batch of inputs using local model for quality assessment
    """
    global _alignment_clf, _embedding_model, _local_model, _local_tokenizer, _generation_kwargs, _weights, _initialized
    
    if not _initialized:
        raise RuntimeError("Reward system not initialized. Call init_shared_reward_system() first.")
    
    if not reward_inputs:
        return []
    
    # Normalize weights
    w_dict = {
        'quality': _weights['quality'],
        'alignment': _weights['alignment'], 
        'format': format_weight
    }
    total = w_dict['quality'] + w_dict['alignment'] + w_dict['format']
    w_normalized = {k: v/total for k, v in w_dict.items()}
    
    # Extract responses and answer parts
    responses = [reward_input.get("response", "") for reward_input in reward_inputs]
    answer_parts = [extract_answer_content(response) for response in responses]
    
    # Batch quality scoring - using local model if available
    try:
        if _local_model is not None and _local_tokenizer is not None:
            gen_kwargs = _generation_kwargs or {}
            q_scores = batch_local_quality_score(answer_parts, _local_model, _local_tokenizer, gen_kwargs)
            print(f"🔍 Batch local model Quality scores: {q_scores}")
        else:
            print("⚠️ Local model not available, using length-based fallback")
            # Fallback to length-based scoring
            q_scores = [min(1.0, len(answer) / 100.0) for answer in answer_parts]
    except Exception as e:
        print(f"⚠️ Batch local model quality scoring failed: {e}")
        q_scores = [0.5] * len(reward_inputs)
    
    # Batch format scoring
    f_scores = [format_reward(response) for response in responses]
    low_format_count = sum(1 for score in f_scores if score < 1)
    if low_format_count > 0:
        print(f"格式分数低于1的样本数量：{low_format_count}")
    
    # Batch alignment scoring
    try:
        if _alignment_clf is not None and _embedding_model is not None:
            a_scores = batch_alignment_reward(answer_parts, _alignment_clf, _embedding_model)
        else:
            print("⚠️ Alignment models not available, using default score")
            a_scores = [0.5] * len(reward_inputs)
    except Exception as e:
        print(f"⚠️ Batch alignment scoring failed: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
        # Fallback to simple scoring
        a_scores = [0.5] * len(reward_inputs)
    
    # Compute overall scores
    results = []
    for i in range(len(reward_inputs)):
        overall = w_normalized['quality'] * q_scores[i] + w_normalized['format'] * f_scores[i] + w_normalized['alignment'] * a_scores[i]
        results.append({
            "overall": float(overall),
            "quality": float(q_scores[i]),
            "format": float(f_scores[i]),
            "alignment": float(a_scores[i]),
        })
    
    return results

# Usage example and consistency test:
if __name__ == "__main__":
    # Initialize with local model
    init_shared_reward_system(
        use_local_reward_model=True,
        local_reward_model_path="/path/to/your/local/model",
        local_reward_model_kwargs={
            "device_map": "auto", 
            "torch_dtype": "float16",
            "batch_size": 4  # Enable batch processing
        },
        alignment_model_path="/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt",
        embedding_model_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
        weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
    )
    
    # Test samples for batch vs single consistency
    test_samples = [
        {
            "response": "<think>This requires merging related Q&A about machine learning</think><answer>Q: What is machine learning and how does it work? A: Machine learning is a subset of artificial intelligence that uses algorithms to automatically learn patterns from data without explicit programming.</answer>",
            "response_length": 200,
            "ground_truth": "Standard ML answer",
        },
        {
            "response": "<think>Another ML question</think><answer>Q: What are neural networks? A: Neural networks are computational models inspired by biological neural networks that can learn complex patterns.</answer>",
            "response_length": 150,
            "ground_truth": "Neural network answer",
        },
        {
            "response": "<think>Simple math</think><answer>Q: What is 2+2? A: 4</answer>",
            "response_length": 50,
            "ground_truth": "Math answer",
        },
        {
            "response": "<think>Complex topic</think><answer>Q: Explain quantum computing? A: Quantum computing uses quantum mechanics principles like superposition and entanglement to process information in fundamentally different ways than classical computers.</answer>",
            "response_length": 180,
            "ground_truth": "Quantum answer",
        }
    ]
    
    print(f"\n=== Testing Batch vs Single Inference Consistency ===")
    
    # Test batch processing
    print(f"\n🚀 Testing batch processing (batch_size=4)...")
    batch_scores = compute_score(test_samples)
    
    # Test single processing for comparison
    print(f"\n🔄 Testing single processing for comparison...")
    single_scores = []
    for sample in test_samples:
        single_score = compute_score([sample])[0]  # Process one at a time
        single_scores.append(single_score)
    
    # Compare results
    print(f"\n📊 Consistency Check:")
    print(f"{'Sample':<8} {'Batch':<15} {'Single':<15} {'Diff':<15} {'Match':<8}")
    print("-" * 70)
    
    total_diff = 0
    matches = 0
    
    for i, (batch_score, single_score) in enumerate(zip(batch_scores, single_scores)):
        diff = abs(batch_score['overall'] - single_score['overall'])
        match = diff < 0.01  # Consider identical if difference < 1%
        total_diff += diff
        if match:
            matches += 1
            
        print(f"Sample {i+1:<2} {batch_score['overall']:<15.3f} {single_score['overall']:<15.3f} {diff:<15.3f} {'✅' if match else '❌':<8}")
    
    avg_diff = total_diff / len(test_samples)
    consistency_rate = matches / len(test_samples) * 100
    
    print("-" * 70)
    print(f"Average difference: {avg_diff:.4f}")
    print(f"Consistency rate: {consistency_rate:.1f}% ({matches}/{len(test_samples)})")
    
    if consistency_rate >= 90:
        print("✅ Batch and single inference are highly consistent!")
    elif consistency_rate >= 75:
        print("⚠️  Batch and single inference have some differences")
    else:
        print("❌ Batch and single inference results differ significantly")