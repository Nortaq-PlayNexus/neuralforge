"""Model registry — track all trained and exported models."""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    """Local JSON database for tracking trained/exported models."""

    def __init__(self, db_path: str | None = None, display: Any = None):
        self.display = display
        self.db_path = Path(db_path) if db_path else Path.home() / ".neuralforge" / "registry.json"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    self.models = json.load(f)
            except Exception:
                self.models = {}

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.models, f, indent=2, default=str)

    def register(self, model_info: dict) -> bool:
        name = model_info.get("name")
        if not name:
            if self.display:
                self.display.print_error("Model name is required")
            return False

        entry = {
            "name": name,
            "path": model_info.get("path", ""),
            "metrics": model_info.get("metrics", {}),
            "training_config": model_info.get("training_config", {}),
            "tags": model_info.get("tags", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "notes": model_info.get("notes", ""),
        }

        if name in self.models:
            entry["created_at"] = self.models[name].get("created_at", entry["created_at"])

        self.models[name] = entry
        self._save()

        if self.display:
            self.display.print_success(f"Model '{name}' registered")
        return True

    def list_models(self, tag: str | None = None) -> list[dict]:
        models = list(self.models.values())
        if tag:
            models = [m for m in models if tag in m.get("tags", [])]
        return sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)

    def get_model(self, name: str) -> dict | None:
        return self.models.get(name)

    def compare(self, model_a: str, model_b: str) -> dict | None:
        a = self.models.get(model_a)
        b = self.models.get(model_b)
        if not a or not b:
            if self.display:
                self.display.print_error(f"Model not found: {model_a if not a else model_b}")
            return None

        comparison = {
            "model_a": a,
            "model_b": b,
            "metric_comparison": {},
        }

        all_metrics = set(list(a.get("metrics", {}).keys()) + list(b.get("metrics", {}).keys()))
        for metric in all_metrics:
            val_a = a.get("metrics", {}).get(metric)
            val_b = b.get("metrics", {}).get(metric)
            comparison["metric_comparison"][metric] = {
                model_a: val_a,
                model_b: val_b,
            }

        if self.display:
            self.display.print_step(f"\n--- Comparison: {model_a} vs {model_b} ---")
            for metric, vals in comparison["metric_comparison"].items():
                self.display.print_step(f"  {metric}: {vals[model_a]} | {vals[model_b]}")

        return comparison

    def delete(self, name: str) -> bool:
        if name not in self.models:
            if self.display:
                self.display.print_error(f"Model '{name}' not found in registry")
            return False
        del self.models[name]
        self._save()
        if self.display:
            self.display.print_success(f"Model '{name}' removed from registry")
        return True
