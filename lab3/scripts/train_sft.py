#!/usr/bin/env python3
"""LoRA/DoRA supervised fine-tuning of Qwen2-7B-Instruct on DISC-Med-SFT."""
import argparse
import json
import os
import random
from dataclasses import dataclass

import torch
import transformers
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
ALL_LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((len(features), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)
        attention = torch.zeros((len(features), max_len), dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            labels[i, :n] = torch.tensor(f["labels"], dtype=torch.long)
            attention[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def build_example(tokenizer, user, target, cutoff_len):
    prefix_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("prompt tokenization is not a prefix of the full conversation")
    labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):]
    return ids[:cutoff_len], labels[:cutoff_len], len(ids) > cutoff_len


def build_samples(tokenizer, path, cutoff_len, seed, max_samples, label):
    """Render an Alpaca JSON file into prompt-masked training samples."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    random.Random(seed).shuffle(raw)
    if max_samples:
        raw = raw[: int(max_samples)]

    samples, skipped, truncated = [], 0, 0
    for i, ex in enumerate(raw):
        user = (ex.get("instruction") or "").strip()
        extra = (ex.get("input") or "").strip()
        if extra:
            user = user + "\n" + extra
        target_text = (ex.get("output") or "").strip()
        if not user or not target_text:
            skipped += 1
            continue
        ids, labels, cut = build_example(tokenizer, user, target_text, cutoff_len)
        if len(ids) < 2 or all(label == -100 for label in labels):
            skipped += 1
            continue
        truncated += int(cut)
        samples.append({"input_ids": ids, "labels": labels})
        if (i + 1) % 5000 == 0:
            print(f"[data] prepared {i + 1}/{len(raw)}", flush=True)
    print(f"[data] {label}: kept={len(samples)} skipped={skipped} truncated={truncated}", flush=True)
    return samples, skipped, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default=None, help="base model path (overrides config)")
    ap.add_argument("--data-dir", default=None, help="directory with train.json (overrides config)")
    ap.add_argument("--out-dir", default=None, help="adapter output directory (overrides config)")
    ap.add_argument("--meta-dir", default="results")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["use_dora"] = bool(cfg.get("use_dora", False))
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    fp16 = args.fp16 or bool(cfg.get("fp16", False))
    gradient_checkpointing = args.gradient_checkpointing or bool(cfg.get("gradient_checkpointing", False))
    model_dtype = torch.float16 if fp16 else torch.bfloat16
    model_path = args.model or cfg["model_name_or_path"]
    major = int(transformers.__version__.split(".")[0])
    dtype_kw = "dtype" if major >= 5 else "torch_dtype"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        trust_remote_code=True,
        device_map={"": 0},
        **{dtype_kw: model_dtype},
    )
    model.config.use_cache = False

    target = cfg.get("lora_target", "all")
    targets = ALL_LINEARS if target == "all" else [t.strip() for t in target.split(",") if t.strip()]
    lora_config = LoraConfig(
        r=int(cfg.get("lora_rank", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        use_dora=cfg["use_dora"],
    )
    model = get_peft_model(model, lora_config)
    if fp16:
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable={trainable} total={total} pct={100 * trainable / total:.4f}%", flush=True)
    print(f"[params] targets={targets} dora={cfg['use_dora']} dtype={model_dtype}", flush=True)

    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    data_dir = args.data_dir or cfg["dataset_dir"]
    cutoff_len = int(cfg.get("cutoff_len", 1024))
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples")
    samples, skipped, truncated = build_samples(
        tokenizer, os.path.join(data_dir, "train.json"), cutoff_len, seed, max_samples, "train")

    eval_samples = None
    val_path = os.path.join(data_dir, "val.json")
    if os.path.exists(val_path):
        eval_samples, _, _ = build_samples(
            tokenizer, val_path, cutoff_len, seed, cfg.get("eval_max_samples", 500), "val")
    else:
        print(f"[data] no {val_path}; training without validation", flush=True)

    out_dir = args.out_dir or cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 8)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1.0)),
        max_steps=args.max_steps,
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        fp16=fp16,
        bf16=not fp16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=int(os.environ.get("DATALOADER_WORKERS", "2")),
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SFTDataset(samples),
        data_collator=PadCollator(tokenizer.pad_token_id),
        **{("processing_class" if major >= 5 else "tokenizer"): tokenizer},
    )
    trainer.train()
    print(f"[gpu] peak_alloc_gb={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)

    eval_loss = None
    if eval_samples:
        eval_out = trainer.evaluate(eval_dataset=SFTDataset(eval_samples), metric_key_prefix="eval")
        eval_loss = eval_out.get("eval_loss")
        print(f"[eval] val_loss={eval_loss:.4f} val_samples={len(eval_samples)}", flush=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    trainer.save_state()

    os.makedirs(args.meta_dir, exist_ok=True)
    meta = {
        "tag": args.tag,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
        "lora_rank": cfg.get("lora_rank"),
        "lora_alpha": cfg.get("lora_alpha"),
        "lora_target": targets,
        "use_dora": cfg["use_dora"],
        "learning_rate": cfg.get("learning_rate"),
        "num_samples": len(samples),
        "skipped": skipped,
        "truncated": truncated,
        "cutoff_len": cutoff_len,
        "eff_batch_size": int(cfg.get("per_device_train_batch_size", 8)) * int(cfg.get("gradient_accumulation_steps", 4)),
        "val_samples": len(eval_samples) if eval_samples else 0,
        "val_loss": eval_loss,
        "steps": int(trainer.state.global_step),
        "fp16": fp16,
        "gradient_checkpointing": gradient_checkpointing,
    }
    with open(os.path.join(args.meta_dir, f"train_meta_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[done]", json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
