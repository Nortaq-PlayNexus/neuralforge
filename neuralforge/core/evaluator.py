"""Model evaluation harness."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any


class ModelEvaluator:
    def __init__(self, display: Any = None):
        self.display = display

    def evaluate(self, model_name: str, dataset_path: str, metrics: list[str]) -> dict:
        if self.display:
            self.display.print_step(f"Evaluating model: {model_name}")
            self.display.print_step(f"Dataset: {dataset_path}")
            self.display.print_step(f"Metrics: {', '.join(metrics)}")

        dataset = self._load_dataset(dataset_path)
        if not dataset:
            return {"success": False, "error": "Failed to load dataset"}

        results = {"model": model_name, "dataset": dataset_path, "metrics": {}, "samples": len(dataset)}

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            if self.display:
                self.display.print_step("Loading model...")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

            predictions = []
            for i, sample in enumerate(dataset[:100]):
                prompt = sample.get("prompt", sample.get("input", ""))
                if not prompt:
                    continue

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
                pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
                expected = sample.get("expected", sample.get("output", ""))
                predictions.append({"predicted": pred, "expected": expected})

                if self.display and (i + 1) % 10 == 0:
                    self.display.print_step(f"  Evaluated {i+1}/{min(len(dataset), 100)} samples")

            for metric in metrics:
                results["metrics"][metric] = self._compute_metric(metric, predictions)

        except ImportError:
            if self.display:
                self.display.print_step("transformers/torch not installed — using mock evaluation")
            for metric in metrics:
                results["metrics"][metric] = {"value": 0.0, "note": "requires transformers + torch"}

        results["success"] = True
        return results

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
        except Exception:
            pass
        return data

    def _compute_metric(self, metric: str, predictions: list[dict]) -> dict:
        if not predictions:
            return {"value": 0.0}
        if metric == "accuracy":
            correct = sum(1 for p in predictions if p["expected"].lower() in p["predicted"].lower())
            return {"value": round(correct / len(predictions), 4)}
        if metric == "f1":
            return {"value": 0.0, "note": "F1 requires labeled classification data"}
        if metric == "rouge":
            return {"value": 0.0, "note": "Install rouge-score for ROUGE metrics"}
        return {"value": 0.0, "note": f"Unknown metric: {metric}"}
