"""NeuralForge — CLI entry point."""

import argparse
import sys

from neuralforge import __version__
from neuralforge.ui.display import Display


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralforge",
        description="Local-first AI platform — fine-tune, RAG, evaluate, deploy.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Show NeuralForge info and capabilities")

    finetune = sub.add_parser("finetune", help="Fine-tune a model locally")
    finetune.add_argument("--model", type=str, required=True, help="Base model name or path")
    finetune.add_argument("--dataset", type=str, required=True, help="Dataset file (JSONL, CSV)")
    finetune.add_argument("--output", type=str, default="./output", help="Output directory")
    finetune.add_argument("--epochs", type=int, default=3)
    finetune.add_argument("--lr", type=float, default=2e-5)
    finetune.add_argument("--batch-size", type=int, default=4)
    finetune.add_argument("--max-length", type=int, default=512)

    rag = sub.add_parser("rag", help="Build a RAG pipeline")
    rag.add_argument("--docs", type=str, required=True, help="Documents directory")
    rag.add_argument("--query", type=str, help="Query to test the pipeline")
    rag.add_argument("--chunk-size", type=int, default=512)
    rag.add_argument("--overlap", type=int, default=50)
    rag.add_argument("--top-k", type=int, default=5)

    evaluate = sub.add_parser("evaluate", help="Evaluate a model")
    evaluate.add_argument("--model", type=str, required=True, help="Model to evaluate")
    evaluate.add_argument("--dataset", type=str, required=True, help="Test dataset")
    evaluate.add_argument("--metrics", nargs="+", default=["accuracy", "f1"])

    export = sub.add_parser("export", help="Export a model for deployment")
    export.add_argument("--model", type=str, required=True, help="Model to export")
    export.add_argument("--format", choices=["gguf", "onnx", "docker"], default="gguf")
    export.add_argument("--output", type=str, default="./export")

    dataset = sub.add_parser("dataset", help="Dataset management")
    dataset.add_argument("action", choices=["info", "validate", "split", "convert"])
    dataset.add_argument("--input", type=str, required=True)
    dataset.add_argument("--output", type=str)
    dataset.add_argument("--format", choices=["jsonl", "csv", "parquet"])
    dataset.add_argument("--split-ratio", type=float, default=0.8)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    display = Display()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "info":
        display.print_info()
        return 0

    if args.command == "finetune":
        from neuralforge.core.finetuner import LocalFinetuner
        finetuner = LocalFinetuner(display=display)
        result = finetuner.finetune(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        return 0 if result else 1

    if args.command == "rag":
        from neuralforge.core.rag import RAGPipeline
        pipeline = RAGPipeline(display=display)
        if args.query:
            result = pipeline.query(args.query)
            display.print_rag_result(result)
        else:
            pipeline.build(args.docs, chunk_size=args.chunk_size, overlap=args.overlap)
        return 0

    if args.command == "evaluate":
        from neuralforge.core.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(display=display)
        result = evaluator.evaluate(args.model, args.dataset, args.metrics)
        display.print_eval_result(result)
        return 0

    if args.command == "export":
        from neuralforge.core.exporter import ModelExporter
        exporter = ModelExporter(display=display)
        result = exporter.export(args.model, args.format, args.output)
        return 0 if result else 1

    if args.command == "dataset":
        from neuralforge.core.dataset import DatasetManager
        manager = DatasetManager(display=display)
        if args.action == "info":
            manager.info(args.input)
        elif args.action == "validate":
            manager.validate(args.input)
        elif args.action == "split":
            manager.split(args.input, args.output, args.split_ratio)
        elif args.action == "convert":
            manager.convert(args.input, args.output, args.format)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
