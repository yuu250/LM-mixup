# -*- coding: utf-8 -*-
import os
import re
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from sentence_transformers import SentenceTransformer
import regex
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

JSON_RE = regex.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE
)

# ───────────────────────── globals ─────────────────────────
_alignment_clf = None               # torch.nn.Module (MLP)
_embedding_model = None             # SentenceTransformer
_rater = None                       # RewardModelScorer instance
_weights = None                     # RewardWeights
_initialized = False                # flag

_default_quality_threshold = 4      # 已不再使用阈值，将使用连续分数，但保留变量不影响

import torch
from torch import nn

# ───────────────────────── MLP 定义 ─────────────────────────
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

def get_alignment_classifier(ckpt_path: Union[str, bytes]) -> MLPClassifier:
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

# ───────────────────────── utils ─────────────────────────
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

def extract_answer_content(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()

def format_reward(response: str) -> float:
    return 1.0 if re.fullmatch(FORMAT_RE, response or "") else 0.0

def alignment_reward(
    pair: str,
    model: torch.nn.Module,
    class_to_score: Dict[int, float] = {0: 0.0, 1: 0.2, 2: 1.0},
) -> float:
    # 依赖全局 _embedding_model
    features = _embedding_model.encode([pair], normalize_embeddings=True)[0]
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = int(torch.argmax(probs).item())  # 0/1/2

    base = class_to_score.get(pred_label, 0.0)
    return base

# ───────────────────────── Reward Model (DeBERTa) ─────────────────────────
class RewardModelScorer:
    """
    使用 OpenAssistant/reward-model-deberta-v3-large-v2 作为质量评分器：
    - 输入: prompt 与 answer（若无 prompt，则仅用 answer）
    - 输出: 一个连续质量分数 q ∈ [0, 1]，由 logits 经过 sigmoid 得到
    """

    def __init__(
        self,
        model_name: str = "/root/autodl-tmp/EasyR1/ckpt/reward-model-deberta-v3-large-v2",
        use_bnb_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 2048,
    ):
        self.model_name = model_name
        self.max_length = max_length

        if use_bnb_4bit:
            # 对 encoder RM 通常没必要 4bit，但如果你硬件紧张可以打开
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map=device_map,
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def _build_input(self, question: Optional[str], answer_text: str) -> str:
        if question and question.strip():
            return f"Question: {question.strip()}\nAnswer: {answer_text.strip()}"
        else:
            return answer_text.strip()

    def score(self, question: Optional[str], answer_text: str) -> float:
        """
        返回质量分数 (0~1)。内部是对 logits 做 sigmoid。
        注意：这是相对意义的质量倾向分数，建议只在同一 prompt 多候选之间比较或排序。
        """
        text = self._build_input(question, answer_text)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()
            # DeBERTa RM 通常是一个标量 logits
            if logits.numel() != 1:
                # 若是多维，取最后一维或第一个标量，这里做兼容处理
                logits = logits.view(-1)[0]
            q = torch.sigmoid(logits).item()

        # 稳健裁剪
        q = float(max(0.0, min(1.0, q)))
        return q

# ───────────────────────── 评分流水线 ─────────────────────────
def compute_score(reward_input: Dict[str, Any], format_weight: float = 0.1) -> Dict[str, float]:
    """
    计算奖励分数 - Sequential模式，单个样本处理
    Args:
        reward_input: 单个样本的输入字典
        format_weight: 格式权重
    Returns:
        单个样本的分数字典
    """
    if not isinstance(reward_input, dict):
        raise ValueError("qa_mixup reward function requires reward_type=sequential")

    if not _initialized:
        raise RuntimeError(
            "Reward system not initialized. Call init_reward_system(...) once in this process first."
        )

    return compute_single_score(reward_input, format_weight)

def compute_single_score(reward_input: Dict[str, Any], format_weight: float = 0.1) -> Dict[str, float]:
    """
    - quality: 使用 OpenAssistant/reward-model-deberta-v3-large-v2 输出的 sigmoid(logits) ∈ [0,1]
    - format: 由固定规则判定（<think></think><answer></answer>）
    - alignment: 由 MLP 分类器（0/1/2）映射得分
    """
    global _rater, _alignment_clf, _weights

    w = RewardWeights(
        quality=_weights.quality,
        alignment=_weights.alignment,
        format=format_weight
    ).normalize()

    response = reward_input.get("response", "")
    question = reward_input.get("question", None)
    answer_part = extract_answer_content(response)

    # 1) 质量分（连续型 0~1）
    try:
        q_score = _rater.score(question, answer_part)  # already in [0,1]
    except Exception as e:
        print(f"⚠️  质量评分失败，使用默认值: {e}")
        q_score = 0.5

    # 2) 格式分（{0,1}）
    f_score = format_reward(response)

    # 3) 对齐分（离散映射）
    try:
        a_score = alignment_reward(answer_part, _alignment_clf)
    except Exception as e:
        print(f"⚠️  对齐评分失败，使用默认值: {e}")
        a_score = 0.5

    overall = w.quality * q_score + w.format * f_score + w.alignment * a_score

    return {
        "overall": float(overall),
        "quality": float(q_score),
        "format": float(f_score),
        "alignment": float(a_score),
    }

# ───────────────────────── 初始化 ─────────────────────────
def init_reward_system(
    *,
    # 使用 OpenAssistant 的 RM
    rm_model_name: str = "/root/autodl-tmp/EasyR1/ckpt/reward-model-deberta-v3-large-v2",
    # alignment MLP ckpt
    alignment_model_path: str = "/root/autodl-tmp/EasyR1/examples/reward_function/mlp_classifier.pt",
    # embedding (用于 alignment)
    embedding_model_name_or_path: str = "/root/autodl-tmp/EasyR1/ckpt/bge-m3",
    weights: RewardWeights = RewardWeights(),
    # RM 资源设置
    use_bnb_4bit: bool = False,
    device_map: str = "auto",
    torch_dtype=torch.float32,
    max_length: int = 2048,
) -> None:
    """
    初始化：
    - 奖励模型（DeBERTa RM）
    - 对齐分类器（MLP）
    - embedding 模型（SentenceTransformer）
    - 归一化权重
    """
    global _alignment_clf, _embedding_model, _rater, _weights, _initialized

    if _initialized:
        print(f"🔄 PID {os.getpid()}: Reward system已经初始化，跳过")
        return

    print(f"🚀 PID {os.getpid()}: 开始初始化 reward system...")

    try:
        # 1) 初始化 Reward Model (DeBERTa)
        print(f"🔄 PID {os.getpid()}: 正在加载 Reward Model: {rm_model_name} ...")
        _rater = RewardModelScorer(
            model_name=rm_model_name,
            use_bnb_4bit=use_bnb_4bit,
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_length=max_length,
        )
        print(f"✅ PID {os.getpid()}: Reward Model 加载完成")

        # 2) 初始化对齐分类器
        print(f"🔄 PID {os.getpid()}: 正在加载对齐分类器: {alignment_model_path} ...")
        _alignment_clf = get_alignment_classifier(alignment_model_path)
        print(f"✅ PID {os.getpid()}: 对齐分类器加载完成")

        # 3) 初始化 embedding 模型
        print(f"🔄 PID {os.getpid()}: 正在加载 embedding 模型: {embedding_model_name_or_path} ...")
        _embedding_model = SentenceTransformer(embedding_model_name_or_path)
        print(f"✅ PID {os.getpid()}: embedding 模型加载完成")

        _weights = weights.normalize()
        _initialized = True
        print(f"🎉 PID {os.getpid()}: Reward system 初始化完全完成!")
    except Exception as e:
        print(f"❌ PID {os.getpid()}: Reward system 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        raise

# ───────────────────────── demo ─────────────────────────
if __name__ == "__main__":
    # 初始化（无锁版本）
    init_reward_system(
        rm_model_name="OpenAssistant/reward-model-deberta-v3-large-v2",
        alignment_model_path="./models/alignment_mlp.pt",
        embedding_model_name_or_path="/root/autodl-tmp/EasyR1/ckpt/bge-m3",
        weights=RewardWeights(quality=0.6, alignment=0.3, format=0.1),
        use_bnb_4bit=False,  # 一般不建议对 encoder RM 4bit，除非显存很紧
    )

    sample1 = {
        "response": "<think>some reasoning</think><answer>Paris is the capital of France.</answer>",
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
    }
    print(compute_score(sample1, format_weight=0.1))

    sample2 = {
        "response": "bad format",
        "question": "Who wrote Hamlet?",
        "answer": "Shakespeare.",
        "rating_overall_compressed": 3,
    }
    print(compute_score(sample2, format_weight=0.1))