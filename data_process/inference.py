import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

THINK_RE  = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are an expert in data mix-up. Your task is to merge the provided content into a single coherent and high-quality output.\n"
    "You MUST ALWAYS follow this exact format:\n"
    "1. First, provide your reasoning wrapped in <think> tags\n"
    "2. Then, provide your final answer wrapped in <answer> tags\n"
    "NEVER output anything outside these tags.\n\n"
    "Example format:\n"
    "<think>\n[Your step-by-step reasoning about how and why the merge makes sense]\n</think>\n\n"
    "<answer>\n[The final merged content]\n</answer>"
)

def read_processed_from_jsonl(path: Path):
    processed_ids = set()
    num_lines = 0
    if not path.exists():
        return processed_ids, 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            num_lines += 1
            try:
                obj = json.loads(line)
                if "id" in obj and obj["id"] is not None:
                    processed_ids.add(obj["id"])
            except Exception:
                pass
    return processed_ids, num_lines

def load_test_items(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    rows: List[Dict[str, Any]] = []

    def _load_one_json(fp: Path, source: str = ""):
        if not fp.exists():
            print(f"[WARN] file not found: {fp}")
            return
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            rows.append({
                "id":          d.get("id", None),
                "input":       d.get("input", ""),
                "output":      d.get("output", ""),
                "cot":         d.get("cot", ""),
                "instruction": d.get("instruction", ""),
                "source":      d.get("source", source),
            })

    if p.is_file():
        _load_one_json(p, source="single")
    elif p.is_dir():
        spec = {
            "cs":   p / "cs"   / "merged_test_cs_noisy.json",
            "mcq":  p / "mcq"  / "merged_test_mcq_noisy.json",
            "qa":   p / "qa"   / "merged_test_qa_noisy.json",
            "para": p / "para" / "merged_test_para_noisy.json",
            "tfq":  p / "tfq"  / "merged_test_tfq_noisy.json",
        }
        for tag, fp in spec.items():
            _load_one_json(fp, source=tag)
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    if not rows:
        raise ValueError(f"No data loaded from: {path}")

    print(f"[INFO] Loaded {len(rows)} test items.")
    return rows


def build_prompt_with_chat_template(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()

    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    if instruction or inp:
        user_content = f"{instruction}\n{inp}" if (instruction and inp) else (instruction or inp)
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": ""})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_generated(text: str) -> Dict[str, str]:
    cot_match = THINK_RE.search(text)
    ans_match = ANSWER_RE.search(text)
    generated_cot = cot_match.group(1).strip() if cot_match else ""
    generated_output = ans_match.group(1).strip() if ans_match else text.strip()
    return {"generated_cot": generated_cot, "generated_output": generated_output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--do_sample", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--override_chat_template", type=str, default=None)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--resume", type=lambda x: x.lower() == "true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.eos_token is None:
        raise ValueError("This model has no eos_token; cannot safely reuse eos as pad.")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.override_chat_template:
        tokenizer.chat_template = args.override_chat_template

    # Dtype selection
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code
    )
    # NOTE: we did NOT add_special_tokens, so no need to resize embeddings.

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # Data
    items = load_test_items(args.test_json)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_ids, processed_lines = read_processed_from_jsonl(out_path)
    total = len(items)
    
    has_all_ids = all(ex.get("id", None) is not None for ex in items)
    if args.resume and out_path.exists():
        if has_all_ids and len(processed_ids) > 0:
            pending_indices = [i for i, ex in enumerate(items) if ex.get("id") not in processed_ids]
        else:
            pending_indices = list(range(processed_lines, total))
    else:
        pending_indices = list(range(total))
    
    if len(pending_indices) == 0:
        print("[INFO] Nothing to do. All items already processed.")
        return
    
    f = out_path.open("a" if args.resume else "w", encoding="utf-8")
    
    bs = max(1, args.batch_size)
    try:
        with tqdm(total=len(pending_indices), desc="Infer (left-pad, resume)" if args.resume else "Infer (left-pad)", dynamic_ncols=True) as pbar:
            for off in range(0, len(pending_indices), bs):
                idxs = pending_indices[off : off + bs]
                batch = [items[i] for i in idxs]
    
                prompts = [build_prompt_with_chat_template(ex, tokenizer) for ex in batch]
    
                enc = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_tokens
                )
                input_ids = enc["input_ids"].to(model.device)
                attention_mask = enc.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(model.device)
    
                base_len = input_ids.shape[1]
    
                with torch.inference_mode():
                    gen = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature if args.do_sample else None,
                        top_p=args.top_p if args.do_sample else None,
                        repetition_penalty=args.repetition_penalty,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                        use_cache=True
                    )
    
                for i, ex in enumerate(batch):
                    new_ids = gen[i][base_len:]
                    if eos_id is not None:
                        pos = (new_ids == eos_id).nonzero(as_tuple=False)
                        if pos.numel() > 0:
                            new_ids = new_ids[: int(pos[0].item())]
    
                    gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    end_tag = "</answer>"
                    cut_pos = gen_text.lower().find(end_tag)
                    if cut_pos != -1:
                        gen_text = gen_text[:cut_pos + len(end_tag)]
    
                    parsed = parse_generated(gen_text)
    
                    global_idx = idxs[i]
                    rec_id = ex.get("id", global_idx)
    
                    record = {
                        "id": rec_id,
                        "source": ex.get("source", ""),
                        "task_type": ex.get("task_type", ""),
                        "generated_cot": parsed["generated_cot"],
                        "generated_output": parsed["generated_output"],
                        "raw_response": gen_text
                    }
    
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    pbar.update(1)
        print(f"[DONE] Appended {len(pending_indices)} lines to {out_path}" if args.resume else f"[DONE] Wrote {total} lines to {out_path}")
    finally:
        f.close()


if __name__ == "__main__":
    main()
