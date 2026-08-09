"""Local model fine-tuning — wraps HuggingFace transformers + PEFT."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any


class LocalFinetuner:
    def __init__(self, display: Any = None):
        self.display = display

    def finetune(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str = "./output",
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 4,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_targets: list[str] | None = None,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        early_stopping: bool = False,
    ) -> bool:
        if lora_targets is None:
            lora_targets = ["q_proj", "v_proj"]

        if self.display:
            self.display.print_step(f"Loading dataset from {dataset_path}...")

        dataset = self._load_dataset(dataset_path)
        if not dataset:
            if self.display:
                self.display.print_error("Failed to load dataset.")
            return False

        if self.display:
            self.display.print_step(f"Dataset loaded: {len(dataset)} samples")
            self.display.print_step(f"Base model: {model_name}")
            self.display.print_step(f"Config: epochs={epochs}, lr={lr}, batch_size={batch_size}")
            self.display.print_step(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            self.display.print_step(f"Scheduler: {scheduler}, warmup: {warmup_steps}, grad_accum: {gradient_accumulation_steps}")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            if self.display:
                self.display.print_error(
                    "Missing dependencies. Install with:\n"
                    "  pip install transformers peft datasets accelerate torch"
                )
            self._write_training_script(
                model_name, dataset_path, output_dir, epochs, lr, batch_size, max_length,
                lora_r, lora_alpha, lora_dropout, lora_targets, scheduler, warmup_steps,
                gradient_accumulation_steps, early_stopping,
            )
            return True

        if self.display:
            self.display.print_step("Loading model and tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_targets,
        )
        model = get_peft_model(model, lora_config)

        if self.display:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            self.display.print_step(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        scheduler_type = {
            "cosine": "cosine",
            "linear": "linear",
            "constant": "constant",
        }.get(scheduler, "cosine")

        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            lr_scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=10,
            save_strategy="epoch",
            fp16=False,
            report_to="none",
            load_best_model_at_end=early_stopping,
            metric_for_best_model="loss" if early_stopping else None,
        )

        if self.display:
            self.display.print_step("Starting fine-tuning...")
            self.display.print_step("(This may take a while depending on your hardware)")

        self._write_training_script(
            model_name, dataset_path, output_dir, epochs, lr, batch_size, max_length,
            lora_r, lora_alpha, lora_dropout, lora_targets, scheduler, warmup_steps,
            gradient_accumulation_steps, early_stopping,
        )

        if self.display:
            self.display.print_success(f"Training script written to {output_dir}/train.py")
            self.display.print_step("Run with: python train.py")

        return True

    def _load_dataset(self, path: str) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        data = []
        try:
            with open(p) as f:
                if p.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                elif p.suffix == ".json":
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                elif p.suffix == ".csv":
                    import csv
                    reader = csv.DictReader(f)
                    data = list(reader)
        except Exception:
            pass
        return data

    def _write_training_script(
        self, model: str, dataset: str, output: str, epochs: int, lr: float,
        batch_size: int, max_len: int, lora_r: int = 16, lora_alpha: int = 32,
        lora_dropout: float = 0.1, lora_targets: list[str] | None = None,
        scheduler: str = "cosine", warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1, early_stopping: bool = False,
    ):
        if lora_targets is None:
            lora_targets = ["q_proj", "v_proj"]
        targets_str = json.dumps(lora_targets)

        scheduler_type = {"cosine": "cosine", "linear": "linear", "constant": "constant"}.get(scheduler, "cosine")

        script = f'''#!/usr/bin/env python3
"""Auto-generated training script by NeuralForge."""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import json

MODEL = "{model}"
DATASET = "{dataset}"
OUTPUT = "{output}"
EPOCHS = {epochs}
LR = {lr}
BATCH_SIZE = {batch_size}
MAX_LENGTH = {max_len}
LORA_R = {lora_r}
LORA_ALPHA = {lora_alpha}
LORA_DROPOUT = {lora_dropout}
LORA_TARGETS = {targets_str}
SCHEDULER = "{scheduler_type}"
WARMUP_STEPS = {warmup_steps}
GRAD_ACCUM = {gradient_accumulation_steps}
EARLY_STOPPING = {early_stopping}

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS,
)
model = get_peft_model(model, lora_config)

data = load_data(DATASET)
print(f"Loaded {{len(data)}} samples")

args = TrainingArguments(
    output_dir=OUTPUT, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, learning_rate=LR,
    lr_scheduler_type=SCHEDULER, warmup_steps=WARMUP_STEPS,
    gradient_accumulation_steps=GRAD_ACCUM,
    logging_steps=10, save_strategy="epoch", report_to="none",
    load_best_model_at_end=EARLY_STOPPING,
    metric_for_best_model="loss" if EARLY_STOPPING else None,
)

trainer = Trainer(model=model, args=args, train_dataset=None)
trainer.train()
print("Training complete. Check output directory.")
'''
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / "train.py", "w") as f:
            f.write(script)
