import os, re, json, argparse, torch, glob
from functools import partial
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, get_peft_model,
    prepare_model_for_kbit_training
)

def load_json_dataset(path: str, *, seed: int = 42) -> Dataset:
    p = Path(path)
    rows = []

    def _load_one_json(fp: Path, source: str = ""):
        if not fp.exists():
            print(f"[WARN] file not found: {fp}")
            return
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            rows.append({
                "input":       d.get("input", ""),
                "output":      d.get("output", ""),
                "cot":         d.get("cot", ""),
                "instruction": d.get("instruction", ""),
                "source":      source,
            })

    if p.is_dir():
        spec = {
            "cs":   p / "cs"   / "merged_train_cs_noisy.json",
            "mcq":  p / "mcq"  / "merged_train_mcq_noisy.json",
            "qa":   p / "qa"   / "merged_train_qa_noisy.json",
            "para": p / "para" / "merged_train_para_noisy.json",
            "tfq":  p / "tfq"  / "merged_train_tfq_noisy.json",
        }
        for tag, fp in spec.items():
            _load_one_json(fp, source=tag)
        if not rows:
            raise FileNotFoundError(f"No JSON files found under directory: {path}")
    elif p.is_file() and p.suffix == ".json":
        _load_one_json(p, source="single")
        if not rows:
            raise ValueError(f"Empty JSON file: {path}")
    else:
        raise FileNotFoundError(f"Path not found or unsupported format: {path}")

    ds = Dataset.from_list(rows)
    ds = ds.shuffle(seed=seed)
    print(f"[INFO] Loaded dataset size: {len(ds)}")
    return ds

def build_prompt_with_chat_template(ex, tokenizer):
    instruction = ex.get("instruction", "").strip()
    inp         = ex.get("input", "").strip()
    system_prompt = (
        "You are an expert in data mix-up. Your task is to merge the provided content into a single coherent and high-quality output.\n"
        "You MUST ALWAYS follow this exact format:\n"
        "1. First, provide your reasoning wrapped in <think> tags\n"
        "2. Then, provide your final answer wrapped in <answer> tags\n"
        "NEVER output anything outside these tags.\n\n"
        "Example format:\n"
        "<think>\n[Your step-by-step reasoning about how and why the merge makes sense]\n</think>\n\n"
        "<answer>\n[The final merged content]\n</answer>"
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    if instruction or inp:
        user_content = f"{instruction}\n{inp}" if (instruction and inp) else instruction or inp
        messages.append({"role": "user", "content": user_content})

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    cot_sec = f"<think>\n{ex.get('cot','').rstrip()}\n</think>\n\n" if ex.get("cot") else ""
    ans_sec = f"<answer>\n{ex['output'].rstrip()}\n</answer>"
    assistant_response = cot_sec + ans_sec

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return {
        "text": prompt + assistant_response,
        "prompt_length": len(prompt_ids)
    }

def preprocess(tokenizer, max_len, example):
    result = tokenizer(example["text"], truncation=True, max_length=max_len)
    result["prompt_length"] = example.get("prompt_length", 0)
    return result

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        prompt_lengths = [f.pop("prompt_length", 0) for f in features]
        batch = super().__call__(features)
        if "labels" in batch:
            for i, pl in enumerate(prompt_lengths):
                if pl > 0:
                    batch["labels"][i][:pl] = -100
        return batch

def validate_format(example):
    output = example.get("output", "")
    if not output:
        print("Warning: Empty output found")
        return False
    return True

def find_latest_checkpoint(output_dir: str) -> str | None:
    pattern = os.path.join(output_dir, "checkpoint-*")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    def step_num(p: str) -> int:
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1
    candidates = sorted(candidates, key=step_num)
    latest = candidates[-1]
    print(f"[INFO] Found latest checkpoint: {latest}")
    return latest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to a directory (merged set) or a single JSON file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", default="./sft_ckpt")
    parser.add_argument("--task_type", choices=["merge", "instruction"], default="instruction")
    parser.add_argument("--max_len", type=int, default=2400)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--use_lora", type=lambda x: x.lower()=="true", default=False)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="*", default=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    parser.add_argument("--quantization", choices=["none", "8bit"], default="none",
                        help="none: FP16/BF16 + LoRA；8bit: INT8 量化 + LoRA")

    parser.add_argument("--trust_remote_code", type=lambda x: x.lower()=="true", default=False)
    parser.add_argument("--override_chat_template", type=str, default=None)

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint dir to resume from, or 'auto' to resume from the latest checkpoint under output_dir."
    )

    parser.add_argument("--save_strategy", choices=["steps", "epoch"], default="epoch")
    parser.add_argument("--save_steps", type=int, default=500)

    args = parser.parse_args()

    ds = load_json_dataset(args.json_path)
    print(f"Original dataset size: {len(ds)}")
    ds = ds.filter(validate_format)
    print(f"Filtered dataset size: {len(ds)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    added_pad = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        added_pad = True

    prompt_fn = partial(build_prompt_with_chat_template, tokenizer=tokenizer)
    ds = ds.map(prompt_fn)

    ds = ds.map(
        partial(preprocess, tokenizer, args.max_len),
        batched=False,
        remove_columns=["input", "output", "cot", "instruction", "text"]
    )

    collator = CustomDataCollator(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    if args.quantization == "8bit":
        quant_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weights=False
        )
    else:
        quant_cfg = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        quantization_config=quant_cfg
    )

    if added_pad:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        if args.quantization == "8bit":
            model = prepare_model_for_kbit_training(model)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    gradient_ckpt = True if args.use_lora else False
    if gradient_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    optim_name = "paged_adamw_8bit" if args.quantization=="8bit" else "adamw_torch"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=20,
        save_total_limit=1,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        optim=optim_name,
        report_to="none",
        warmup_ratio=0.05,
        weight_decay=0.01,
        gradient_checkpointing=gradient_ckpt,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator
    )

    resume_path = None
    if args.resume_from:
        if args.resume_from.strip().lower() == "auto":
            resume_path = find_latest_checkpoint(args.output_dir)
        else:
            resume_path = args.resume_from
            if not os.path.isdir(resume_path):
                raise FileNotFoundError(f"--resume_from path not found: {resume_path}")

    if resume_path:
        print(f"[INFO] Resuming training from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        auto_ckpt = find_latest_checkpoint(args.output_dir)
        if auto_ckpt:
            print(f"[HINT] Detected existing checkpoints under output_dir. You can pass --resume_from auto to continue from: {auto_ckpt}")
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metadata = {
        "format_instruction": "Output format: <think>reasoning</think>\\n\\n<answer>final answer</answer>",
        "stop_tokens": ["</answer>"],
        "system_prompt_used": True
    }
    with open(os.path.join(args.output_dir, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()

