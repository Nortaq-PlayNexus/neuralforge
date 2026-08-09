"""Dataset management — info, validate, split, convert, augment, analyze."""

from __future__ import annotations
import csv
import hashlib
import json
import random
import re
from collections import Counter
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

    def augment(self, input_path: str, output_path: str, techniques: list[str] | None = None) -> dict:
        if techniques is None:
            techniques = ["synonym", "swap"]

        data = self._load(Path(input_path))
        if not data:
            if self.display:
                self.display.print_error("No data to augment")
            return {"success": False, "error": "No data"}

        augmented = list(data)
        tech_counts = {t: 0 for t in techniques}

        for technique in techniques:
            if technique == "synonym":
                new_rows = [self._synonym_replacement(row) for row in data]
                augmented.extend(new_rows)
                tech_counts["synonym"] = len(new_rows)
            elif technique == "swap":
                new_rows = [self._random_swap(row) for row in data]
                augmented.extend(new_rows)
                tech_counts["swap"] = len(new_rows)
            elif technique == "insert":
                new_rows = [self._random_insertion(row) for row in data]
                augmented.extend(new_rows)
                tech_counts["insert"] = len(new_rows)
            elif technique == "backtranslation":
                new_rows = [self._backtranslation_stub(row) for row in data]
                augmented.extend(new_rows)
                tech_counts["backtranslation"] = len(new_rows)

        out = Path(output_path)
        self._save(augmented, out)

        result = {
            "success": True,
            "original": len(data),
            "augmented": len(augmented),
            "techniques": tech_counts,
            "output": str(out),
        }
        if self.display:
            self.display.print_success(f"Augmented {len(data)} -> {len(augmented)} samples")
            for t, c in tech_counts.items():
                self.display.print_step(f"  {t}: +{c}")
        return result

    def analyze(self, path: str) -> dict:
        data = self._load(Path(path))
        if not data:
            return {"error": "No data loaded"}

        stats = {
            "total_samples": len(data),
            "file_size_bytes": Path(path).stat().st_size if Path(path).exists() else 0,
        }

        if data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            stats["fields"] = keys

            null_counts = {k: 0 for k in keys}
            for row in data:
                for k in keys:
                    if row.get(k) is None or row.get(k) == "":
                        null_counts[k] += 1
            stats["null_fields"] = null_counts

            length_field = None
            for candidate in ["text", "content", "input", "prompt", "output", "expected"]:
                if candidate in keys:
                    length_field = candidate
                    break

            if length_field:
                lengths = [len(str(row.get(length_field, ""))) for row in data]
                stats["avg_length"] = round(sum(lengths) / len(lengths), 1) if lengths else 0
                stats["min_length"] = min(lengths) if lengths else 0
                stats["max_length"] = max(lengths) if lengths else 0

            label_field = None
            for candidate in ["label", "category", "class", "target", "expected"]:
                if candidate in keys:
                    label_field = candidate
                    break

            if label_field:
                labels = Counter(str(row.get(label_field, "")) for row in data)
                stats["class_distribution"] = dict(labels.most_common(20))
                stats["num_classes"] = len(labels)

        if self.display:
            self.display.print_step(f"Total samples: {stats['total_samples']}")
            if "fields" in stats:
                self.display.print_step(f"Fields: {', '.join(stats['fields'])}")
            if "class_distribution" in stats:
                self.display.print_step(f"Classes: {stats['num_classes']}")
                for label, count in list(stats["class_distribution"].items())[:10]:
                    self.display.print_step(f"  {label}: {count}")
            if "avg_length" in stats:
                self.display.print_step(f"Avg length: {stats['avg_length']} chars")

        return stats

    def deduplicate(self, input_path: str, output_path: str | None = None, threshold: float = 0.9) -> dict:
        data = self._load(Path(input_path))
        if not data:
            return {"error": "No data"}

        seen_exact: set[str] = set()
        unique = []
        duplicates = 0
        near_duplicates = 0

        for row in data:
            row_str = json.dumps(row, sort_keys=True)
            row_hash = hashlib.md5(row_str.encode()).hexdigest()

            if row_hash in seen_exact:
                duplicates += 1
                continue

            is_near_dup = False
            text = self._get_text_content(row)
            if text:
                text_lower = text.lower()
                for existing_text in [self._get_text_content(u) for u in unique[-100:]]:
                    if existing_text:
                        sim = self._text_similarity(text_lower, existing_text.lower())
                        if sim >= threshold:
                            near_duplicates += 1
                            is_near_dup = True
                            break

            if not is_near_dup:
                seen_exact.add(row_hash)
                unique.append(row)

        out_path = Path(output_path) if output_path else Path(input_path)
        self._save(unique, out_path)

        result = {
            "original": len(data),
            "unique": len(unique),
            "exact_duplicates": duplicates,
            "near_duplicates": near_duplicates,
            "output": str(out_path),
        }
        if self.display:
            self.display.print_success(f"Deduplication: {len(data)} -> {len(unique)} samples")
            self.display.print_step(f"  Exact duplicates removed: {duplicates}")
            self.display.print_step(f"  Near-duplicates removed: {near_duplicates}")
        return result

    def _get_text_content(self, row: dict) -> str:
        for key in ["text", "content", "input", "prompt", "output", "expected"]:
            if key in row and row[key]:
                return str(row[key])
        return ""

    def _text_similarity(self, a: str, b: str) -> float:
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    def _synonym_replacement(self, row: dict) -> dict:
        new_row = dict(row)
        text_key = self._find_text_key(new_row)
        if not text_key:
            return new_row
        text = new_row[text_key]
        words = text.split()
        if not words:
            return new_row
        idx = random.randint(0, len(words) - 1)
        word = words[idx].lower()
        synonyms = {
            "good": ["great", "excellent", "fine"],
            "bad": ["poor", "terrible", "awful"],
            "happy": ["glad", "pleased", "delighted"],
            "sad": ["unhappy", "sorrowful", "down"],
            "fast": ["quick", "rapid", "swift"],
            "slow": ["sluggish", "gradual", "unhurried"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"],
            "important": ["crucial", "vital", "significant"],
            "help": ["assist", "support", "aid"],
        }
        if word in synonyms:
            replacement = random.choice(synonyms[word])
            words[idx] = replacement
        new_row[text_key] = " ".join(words)
        return new_row

    def _random_swap(self, row: dict) -> dict:
        new_row = dict(row)
        text_key = self._find_text_key(new_row)
        if not text_key:
            return new_row
        words = new_row[text_key].split()
        if len(words) < 2:
            return new_row
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
        new_row[text_key] = " ".join(words)
        return new_row

    def _random_insertion(self, row: dict) -> dict:
        new_row = dict(row)
        text_key = self._find_text_key(new_row)
        if not text_key:
            return new_row
        words = new_row[text_key].split()
        if not words:
            return new_row
        idx = random.randint(0, len(words) - 1)
        insert_word = random.choice(words)
        words.insert(idx, insert_word)
        new_row[text_key] = " ".join(words)
        return new_row

    def _backtranslation_stub(self, row: dict) -> dict:
        new_row = dict(row)
        text_key = self._find_text_key(new_row)
        if not text_key:
            return new_row
        words = new_row[text_key].split()
        if not words:
            return new_row
        simplified = []
        for w in words:
            if len(w) > 6:
                simplified.append(w[:len(w)//2])
            else:
                simplified.append(w)
        new_row[text_key] = " ".join(simplified)
        return new_row

    def _find_text_key(self, row: dict) -> str | None:
        for key in ["text", "content", "input", "prompt", "output", "expected"]:
            if key in row:
                return key
        return None

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
