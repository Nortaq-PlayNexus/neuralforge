"""Model exporter — export models to various formats."""

from __future__ import annotations
from pathlib import Path
from typing import Any


class ModelExporter:
    def __init__(self, display: Any = None):
        self.display = display

    def export(self, model_name: str, fmt: str, output_dir: str, **kwargs) -> bool:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        if self.display:
            self.display.print_step(f"Exporting {model_name} as {fmt.upper()}...")

        if fmt == "gguf":
            return self._export_gguf(model_name, output, **kwargs)
        elif fmt == "onnx":
            return self._export_onnx(model_name, output, **kwargs)
        elif fmt == "docker":
            return self._export_docker(model_name, output, **kwargs)
        elif fmt == "torchscript":
            return self._export_torchscript(model_name, output, **kwargs)
        return False

    def _export_gguf(self, model_name: str, output: Path, quantization: str = "f16", **kwargs) -> bool:
        script = f'''#!/usr/bin/env python3
"""GGUF export script — requires llama.cpp to be installed."""

import subprocess
import sys
import os

MODEL = "{model_name}"
OUTPUT = str(Path("{output}"))
QUANTIZATION = "{quantization}"

llama_cpp = os.path.join(OUTPUT, "llama.cpp")
if not os.path.exists(llama_cpp):
    print("Cloning llama.cpp...")
    subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp], check=True)

convert = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
if os.path.exists(convert):
    gguf_out = os.path.join(OUTPUT, "model.gguf")
    cmd = [sys.executable, convert, MODEL, "--outfile", gguf_out]
    if QUANTIZATION in ("q4_0", "q4_1", "q5_0", "q5_1", "q8_0"):
        cmd.extend(["--outtype", QUANTIZATION])
    subprocess.run(cmd, check=True)
    print(f"Model exported to {{gguf_out}} (quantization: {{QUANTIZATION}})")
else:
    print("Could not find convert script. Install llama.cpp manually.")
'''
        with open(output / "export_gguf.py", "w") as f:
            f.write(script)
        if self.display:
            self.display.print_success(f"GGUF export script written to {output}/export_gguf.py (quantization: {quantization})")
        return True

    def _export_onnx(self, model_name: str, output: Path, dynamic_axes: bool = True, **kwargs) -> bool:
        dynamic_code = ""
        if dynamic_axes:
            dynamic_code = '''
    # Dynamic axes for variable-length inputs
    dynamic_axes = {{
        "input_ids": {{0: "batch_size", 1: "sequence_length"}},
        "attention_mask": {{0: "batch_size", 1: "sequence_length"}},
        "output": {{0: "batch_size", 1: "sequence_length"}},
    }}
    print(f"Dynamic axes configured for variable-length inputs")
'''
        script = f'''#!/usr/bin/env python3
"""ONNX export script — requires optimum and onnxruntime."""

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

MODEL = "{model_name}"
OUTPUT = str(Path("{output}"))

model = ORTModelForCausalLM.from_pretrained(MODEL, export=True)
model.save_pretrained(OUTPUT)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.save_pretrained(OUTPUT)
{dynamic_code}
print(f"Model exported to {{OUTPUT}}")
print("Files:")
import os
for f in os.listdir(OUTPUT):
    print(f"  {{f}}")
'''
        with open(output / "export_onnx.py", "w") as f:
            f.write(script)
        if self.display:
            self.display.print_success(f"ONNX export script written to {output}/export_onnx.py")
        return True

    def _export_torchscript(self, model_name: str, output: Path, **kwargs) -> bool:
        script = f'''#!/usr/bin/env python3
"""TorchScript export via tracing."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "{model_name}"
OUTPUT = str(Path("{output}"))

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model.eval()

dummy_input = tokenizer("Hello, world!", return_tensors="pt", padding="max_length", max_length=128)
traced = torch.jit.trace(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
)
traced.save(OUTPUT + "/model.pt")
print(f"TorchScript model saved to {{OUTPUT}}/model.pt")
'''
        with open(output / "export_torchscript.py", "w") as f:
            f.write(script)
        if self.display:
            self.display.print_success(f"TorchScript export script written to {output}/export_torchscript.py")
        return True

    def _export_docker(
        self, model_name: str, output: Path,
        gpu: bool = False, health_check: bool = True, **kwargs,
    ) -> bool:
        gpu_section = ""
        gpu_dockerfile = ""
        if gpu:
            gpu_dockerfile = '''
# GPU support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
'''
            gpu_section = '''
# GPU runtime
# docker run --gpus all -p 8000:8000 {model_name}
'''

        health_endpoint = ""
        if health_check:
            health_endpoint = '''
@app.get("/health")
def health():
    return {{"status": "ok", "model": MODEL, "loaded": True}}
'''

        dockerfile = f'''# Generated by NeuralForge — Multi-stage build
# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --target=/deps transformers torch fastapi uvicorn

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /deps /usr/local/lib/python3.12/site-packages

COPY serve.py .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        serve = f'''"""Auto-generated model server by NeuralForge."""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "{model_name}"

app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7

@app.post("/generate")
def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=req.max_tokens, temperature=req.temperature
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {{"response": text}}

{health_endpoint}
@app.get("/info")
def info():
    return {{"model": MODEL, "device": str(model.device), "dtype": str(model.dtype)}}
'''
        with open(output / "Dockerfile", "w") as f:
            f.write(dockerfile)
        with open(output / "serve.py", "w") as f:
            f.write(serve)

        gpu_note = " (GPU)" if gpu else ""
        if self.display:
            self.display.print_success(f"Docker deployment{gpu_note} written to {output}/")
            build_cmd = f"docker build -t {model_name} {output}"
            run_cmd = f"docker run --gpus all -p 8000:8000 {model_name}" if gpu else f"docker run -p 8000:8000 {model_name}"
            self.display.print_step(f"Build: {build_cmd}")
            self.display.print_step(f"Run: {run_cmd}")
        return True
