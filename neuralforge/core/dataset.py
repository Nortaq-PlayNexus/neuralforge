"""Dataset management — info, validate, split, convert."""

from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Any


class DatasetManager:
    def __init__(self, display: Any = None):
        self.display = display

    def info(self, path: str):
        p = Path(path)
        if not p.exists():
            if self.display:
                self.display.print_error(f"File not found: {path}")
            return

        data = self._load(p)
        if self.display:
            self.display.print_step(f"File: {p.name}")
            self.display.print_step(f"Format: {p.suffix}")
            self.display.print_step(f"Size: {p.stat().st_size:,} bytes")
            self.display.print_step(f"Samples: {len(data)}")
            if data:
                keys = list(data[0].keys()) if isinstance(data[0], dict) else []
                if keys:
                    self.display.print_step(f"Fields: {', '.join(keys)}")
                sample = data[0]
                if isinstance(sample, dict):
                    self.display.print_step(f"Sample: {json.dumps(sample, indent=2)[:500]}")

    def validate(self, path: str):
        p = Path(path)
        data = self._load(p)
        issues = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                issues.append(f"Row {i}: not a dict")
                continue
            if not row:
                issues.append(f"Row {i}: empty")
            for k, v in row.items():
                if v is None:
                    issues.append(f"Row {i}: field '{k}' is None")

        if self.display:
            if issues:
                self.display.print_error(f"Found {len(issues)} issues:")
                for issue in issues[:20]:
                    self.display.print_step(f"  {issue}")
            else:
                self.display.print_success(f"Dataset is valid ({len(data)} samples, 0 issues)")

    def split(self, input_path: str, output_path: str | None, ratio: float = 0.8):
        data = self._load(Path(input_path))
        split_idx = int(len(data) * ratio)
        train = data[:split_idx]
        test = data[split_idx:]

        base = Path(output_path) if output_path else Path(input_path).stem
        out_dir = Path(f"{base}_split")
        out_dir.mkdir(parents=True, exist_ok=True)

        self._save(train, out_dir / "train.jsonl")
        self._save(test, out_dir / "test.jsonl")

        if self.display:
            self.display.print_success(f"Split {len(data)} samples -> train={len(train)}, test={len(test)}")
            self.display.print_step(f"Saved to {out_dir}/")

    def convert(self, input_path: str, output_path: str | None, fmt: str):
        data = self._load(Path(input_path))
        out = Path(output_path) if output_path else Path(input_path).with_suffix(f".{fmt}")
        self._save(data, out)
        if self.display:
            self.display.print_success(f"Converted {len(data)} samples to {fmt}: {out}")

    def _load(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            with open(path) as f:
                if path.suffix == ".jsonl":
                    return [json.loads(line) for line in f if line.strip()]
                elif path.suffix == ".json":
                    data = json.load(f)
                    return data if isinstance(data, list) else [data]
                elif path.suffix == ".csv":
                    return list(csv.DictReader(f))
        except Exception:
            pass
        return []

    def _save(self, data: list[dict], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            if path.suffix == ".jsonl":
                for row in data:
                    f.write(json.dumps(row) + "\n")
            else:
                json.dump(data, f, indent=2)
