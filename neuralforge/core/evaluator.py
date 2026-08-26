"""Model evaluation harness."""

from __future__ import annotations
import json
import math
from collections import Counter
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

        results = {
            "model": model_name,
            "dataset": dataset_path,
            "metrics": {},
            "samples": len(dataset),
        }

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            if self.display:
                self.display.print_step("Loading model...")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

            predictions = []
            all_logits = []
            for i, sample in enumerate(dataset[:100]):
                prompt = sample.get("prompt", sample.get("input", ""))
                if not prompt:
                    continue

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        return_dict_in_generate=True,
                    )
                    logits = outputs.logits if hasattr(outputs, "logits") else None
                pred = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                expected = sample.get("expected", sample.get("output", ""))
                predictions.append({"predicted": pred, "expected": expected})
                if logits is not None:
                    all_logits.append(logits)

                if self.display and (i + 1) % 10 == 0:
                    self.display.print_step(f"  Evaluated {i + 1}/{min(len(dataset), 100)} samples")

            for metric in metrics:
                if metric == "perplexity" and all_logits:
                    results["metrics"][metric] = self._compute_perplexity(all_logits)
                else:
                    results["metrics"][metric] = self._compute_metric(metric, predictions)

        except ImportError:
            if self.display:
                self.display.print_step("transformers/torch not installed — using mock evaluation")
            for metric in metrics:
                results["metrics"][metric] = {
                    "value": 0.0,
                    "note": "requires transformers + torch",
                }

        results["success"] = True
        return results

    def compare_models(
        self,
        model_a: str,
        model_b: str,
        dataset_path: str,
        metrics: list[str] | None = None,
    ) -> dict:
        if metrics is None:
            metrics = ["accuracy", "f1", "bleu", "rouge"]
        if self.display:
            self.display.print_step(f"Comparing {model_a} vs {model_b}")

        result_a = self.evaluate(model_a, dataset_path, metrics)
        result_b = self.evaluate(model_b, dataset_path, metrics)

        comparison = {
            "model_a": {"name": model_a, "metrics": result_a.get("metrics", {})},
            "model_b": {"name": model_b, "metrics": result_b.get("metrics", {})},
            "dataset": dataset_path,
            "samples": result_a.get("samples", 0),
        }

        if self.display:
            self.display.print_step("\n--- Model Comparison ---")
            for m in metrics:
                val_a = comparison["model_a"]["metrics"].get(m, {}).get("value", "N/A")
                val_b = comparison["model_b"]["metrics"].get(m, {}).get("value", "N/A")
                self.display.print_step(f"  {m}: {model_a}={val_a} | {model_b}={val_b}")

        return comparison

    def export_results(self, results: dict, output_path: str):
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        if self.display:
            self.display.print_success(f"Results exported to {output_path}")

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
            return self._compute_f1(predictions)
        if metric == "bleu":
            return self._compute_bleu(predictions)
        if metric == "rouge":
            return self._compute_rouge_l(predictions)
        if metric == "confusion_matrix":
            return self._compute_confusion_matrix(predictions)
        return {"value": 0.0, "note": f"Unknown metric: {metric}"}

    def _compute_bleu(self, predictions: list[dict]) -> dict:
        """Compute BLEU score from scratch using n-gram precision."""
        if not predictions:
            return {"value": 0.0}

        total_score = 0.0
        for pred in predictions:
            ref_tokens = pred["expected"].lower().split()
            hyp_tokens = pred["predicted"].lower().split()
            if not ref_tokens or not hyp_tokens:
                continue

            clipped_counts = 0
            total_counts = 0
            for n in range(1, 5):
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                hyp_ngrams = self._get_ngrams(hyp_tokens, n)
                for ngram, count in hyp_ngrams.items():
                    clipped_counts += min(count, ref_ngrams.get(ngram, 0))
                    total_counts += count

            if total_counts == 0:
                continue

            precision = clipped_counts / total_counts
            brevity_penalty = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
            total_score += brevity_penalty * precision

        avg_score = total_score / len(predictions) if predictions else 0.0
        return {"value": round(avg_score, 4)}

    def _get_ngrams(self, tokens: list[str], n: int) -> Counter:
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams[ngram] += 1
        return ngrams

    def _compute_f1(self, predictions: list[dict]) -> dict:
        """Compute macro F1 for classification-style predictions."""
        tp = fp = fn = 0
        for pred in predictions:
            expected = pred["expected"].strip().lower()
            predicted = pred["predicted"].strip().lower()
            if expected == predicted:
                tp += 1
            else:
                fp += 1
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "value": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    def _compute_rouge_l(self, predictions: list[dict]) -> dict:
        """Compute word-level ROUGE-L using LCS."""
        if not predictions:
            return {"value": 0.0}

        total_f1 = 0.0
        for pred in predictions:
            ref = pred["expected"].lower().split()
            hyp = pred["predicted"].lower().split()
            lcs_len = self._lcs_length(ref, hyp)
            precision = lcs_len / len(hyp) if hyp else 0.0
            recall = lcs_len / len(ref) if ref else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            total_f1 += f1

        return {"value": round(total_f1 / len(predictions), 4) if predictions else 0.0}

    def _lcs_length(self, a: list[str], b: list[str]) -> int:
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return 0
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def _compute_perplexity(self, logits_list: list) -> dict:
        """Compute perplexity from logits."""
        total_loss = 0.0
        total_tokens = 0
        for logits in logits_list:
            if logits is None or logits.dim() < 3:
                continue
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = logits[..., 1:, :].contiguous()
            import torch

            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item() * shift_logits.size(0)
            total_tokens += shift_logits.size(0)

        if total_tokens == 0:
            return {"value": 0.0, "note": "No valid logits"}
        avg_loss = total_loss / total_tokens
        return {"value": round(math.exp(avg_loss), 4)}

    def _compute_confusion_matrix(self, predictions: list[dict]) -> dict:
        """Generate a confusion matrix for classification."""
        labels = sorted(
            set(p["expected"] for p in predictions) | set(p["predicted"] for p in predictions)
        )
        matrix = {l: {l2: 0 for l2 in labels} for l in labels}
        for pred in predictions:
            expected = pred["expected"].strip().lower()
            predicted = pred["predicted"].strip().lower()
            if expected in matrix and predicted in matrix[expected]:
                matrix[expected][predicted] += 1
        return {"labels": labels, "matrix": matrix}
