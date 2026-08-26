"""Rich CLI display for NeuralForge."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


console = Console()


class Display:
    def print_step(self, message: str):
        console.print(f"  [bold blue]→[/bold blue] {message}")

    def print_success(self, message: str):
        console.print(f"  [bold green]✓[/bold green] {message}")

    def print_error(self, message: str):
        console.print(f"  [bold red]✗[/bold red] {message}")

    def print_rag_result(self, result: dict):
        console.print()
        console.print(f"[bold]Query:[/bold] {result.get('query', '')}")
        console.print()
        for i, r in enumerate(result.get("results", []), 1):
            console.print(
                f"  [{i}] score={r.get('score', 0)} source={r.get('source', '?')}"
            )
            text = r.get("text", "")[:200]
            console.print(f"      {text}...")
        console.print()

    def print_eval_result(self, result: dict):
        console.print()
        table = Table(box=box.ROUNDED, title="Evaluation Results")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")
        for metric, val in result.get("metrics", {}).items():
            if isinstance(val, dict):
                table.add_row(metric, str(val.get("value", "N/A")))
            else:
                table.add_row(metric, str(val))
        table.add_row("Samples", str(result.get("samples", 0)))
        console.print(table)
        console.print()

    def print_info(self):
        console.print()
        console.print(
            Panel(
                "[bold cyan]NeuralForge[/bold cyan] — Local-First AI Platform\n\n"
                "Fine-tune, build RAG pipelines, evaluate, and deploy — entirely on your hardware.\n\n"
                "[bold]Commands:[/bold]\n"
                "  neuralforge finetune --model <model> --dataset <file>    Fine-tune a model\n"
                "  neuralforge rag --docs <dir> --query <text>              Build/test RAG pipeline\n"
                "  neuralforge evaluate --model <model> --dataset <file>    Evaluate a model\n"
                "  neuralforge export --model <model> --format gguf         Export for deployment\n"
                "  neuralforge dataset info <file>                         Dataset info\n"
                "  neuralforge dataset validate <file>                     Validate dataset\n"
                "  neuralforge dataset split <file>                        Train/test split\n"
                "  neuralforge dataset convert <file> --format jsonl        Format conversion\n\n"
                "[bold]Supported formats:[/bold]  JSONL, CSV, JSON\n"
                "[bold]Export formats:[/bold]      GGUF, ONNX, Docker\n"
                "[bold]Requirements:[/bold]        Python 3.11+, optional: transformers, torch, peft",
                border_style="blue",
                expand=True,
            )
        )
        console.print()
