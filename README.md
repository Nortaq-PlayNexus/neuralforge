<div align="center">

# NeuralForge

**Local-first AI platform — fine-tune, build RAG pipelines, evaluate, and deploy entirely on your hardware.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-3776AB?logo=python&logoColor=white)](https://python.org)

</div>

**NeuralForge** is a local-first AI development platform that handles the full lifecycle: dataset management, model fine-tuning, RAG pipeline construction, evaluation, and export — all running on your own hardware with zero cloud dependency.

No API costs. No data leaving your machine. Full control.

---

\n---\n\n## Screenshots\n\n| Preview | Description |\n|---------|-------------|\n| ![screenshot](docs/screenshots/screenshot.png) | Main interface |\n| ![screenshot](docs/screenshots/demo.gif) | Demo |\n\n*Screenshots coming soon — placeholders auto-generated. Replace docs/screenshots/ with real captures.*\n\n## Features

### Fine-tuning
- One-command fine-tuning with LoRA/PEFT
- Auto-generates training scripts for HuggingFace transformers
- Supports JSONL, CSV, and JSON datasets
- Configurable epochs, learning rate, batch size

### RAG pipelines
- Document ingestion (txt, md, py, json, yaml)
- Configurable chunk size and overlap
- TF-IDF based retrieval (no external embedding service needed)
- Query interface for testing

### Evaluation
- Multi-metric evaluation (accuracy, F1, ROUGE)
- Uses local model inference
- Structured evaluation reports

### Export
- GGUF (via llama.cpp conversion)
- ONNX (via optimum)
- Docker deployment (FastAPI server with one command)

### Dataset management
- Dataset info and validation
- Train/test split
- Format conversion (JSONL, CSV, JSON)

---

## Quick start

```bash
# Install
pip install -e .

# Fine-tune a model
neuralforge finetune --model distilgpt2 --dataset data/train.jsonl

# Build a RAG pipeline
neuralforge rag --docs ./docs --query "What is the API?"

# Evaluate a model
neuralforge evaluate --model ./output --dataset data/test.jsonl

# Export as Docker deployment
neuralforge export --model ./output --format docker

# Dataset management
neuralforge dataset info data/train.jsonl
neuralforge dataset split data/all.jsonl --split-ratio 0.8
```

---

## Commands

| Command | Description |
|---|---|
| `neuralforge finetune` | Fine-tune a model locally with LoRA |
| `neuralforge rag` | Build and query a RAG pipeline |
| `neuralforge evaluate` | Evaluate model performance |
| `neuralforge export` | Export model (GGUF, ONNX, Docker) |
| `neuralforge dataset` | Dataset management tools |
| `neuralforge info` | Show capabilities and requirements |

---

## How it works

```
Dataset (JSONL/CSV/JSON)
         │
         ▼
┌─────────────────────────────────────┐
│           NeuralForge               │
├─────────────────────────────────────┤
│  Fine-tuner  │  RAG  │  Evaluator  │
│  Dataset Mgr │       │  Exporter   │
└──────────────┴───────┴─────────────┘
         │
         ├──► Fine-tuned model (local)
         ├──► RAG index (local)
         ├──► Evaluation report
         └──► Export (GGUF / ONNX / Docker)
```

---

## Requirements

- Python 3.11+
- Optional: `transformers`, `peft`, `datasets`, `torch` (for fine-tuning/eval)
- 8GB+ RAM recommended
- GPU recommended for fine-tuning

---

## License

[MIT](LICENSE) — PlayNexus
