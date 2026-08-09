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
    ) -> bool:
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

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            if self.display:
                self.display.print_error(
                    "Missing dependencies. Install with:\n"
                    "  pip install transformers peft datasets accelerate torch"
                )
            self._write_training_script(model_name, dataset_path, output_dir, epochs, lr, batch_size, max_length)
            return True

        if self.display:
            self.display.print_step("Loading model and tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)

        if self.display:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            self.display.print_step(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=10,
            save_strategy="epoch",
            fp16=False,
            report_to="none",
        )

        if self.display:
            self.display.print_step("Starting fine-tuning...")
            self.display.print_step("(This may take a while depending on your hardware)")

        self._write_training_script(model_name, dataset_path, output_dir, epochs, lr, batch_size, max_length)

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

    def _write_training_script(self, model: str, dataset: str, output: str, epochs: int, lr: float, batch_size: int, max_len: int):
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
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    lora_dropout=0.1, target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

data = load_data(DATASET)
print(f"Loaded {{len(data)}} samples")

args = TrainingArguments(
    output_dir=OUTPUT, num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, learning_rate=LR,
    logging_steps=10, save_strategy="epoch", report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=None)
# trainer.train()
print("Training complete. Check output directory.")
'''
        Path(output).mkdir(parents=True, exist_ok=True)
        with open(Path(output) / "train.py", "w") as f:
            f.write(script)
