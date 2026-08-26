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
    finetune.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    finetune.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    finetune.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    finetune.add_argument(
        "--lora-targets",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="LoRA target modules",
    )
    finetune.add_argument(
        "--scheduler",
        choices=["cosine", "linear", "constant"],
        default="cosine",
        help="LR scheduler",
    )
    finetune.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps")
    finetune.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    finetune.add_argument("--early-stopping", action="store_true", help="Enable early stopping")

    rag = sub.add_parser("rag", help="Build a RAG pipeline")
    rag.add_argument("--docs", type=str, required=True, help="Documents directory")
    rag.add_argument("--query", type=str, help="Query to test the pipeline")
    rag.add_argument("--chunk-size", type=int, default=512)
    rag.add_argument("--overlap", type=int, default=50)
    rag.add_argument("--top-k", type=int, default=5)
    rag.add_argument("--save-index", type=str, help="Save index to path")
    rag.add_argument("--load-index", type=str, help="Load index from path")

    evaluate = sub.add_parser("evaluate", help="Evaluate a model")
    evaluate.add_argument("--model", type=str, required=True, help="Model to evaluate")
    evaluate.add_argument("--dataset", type=str, required=True, help="Test dataset")
    evaluate.add_argument("--metrics", nargs="+", default=["accuracy", "f1"])
    evaluate.add_argument("--export-results", type=str, help="Export results to JSON file")
    evaluate.add_argument("--compare-with", type=str, help="Second model for comparison")

    export = sub.add_parser("export", help="Export a model for deployment")
    export.add_argument("--model", type=str, required=True, help="Model to export")
    export.add_argument(
        "--format", choices=["gguf", "onnx", "docker", "torchscript"], default="gguf"
    )
    export.add_argument("--output", type=str, default="./export")
    export.add_argument(
        "--quantization",
        choices=["f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
        default="f16",
        help="GGUF quantization",
    )
    export.add_argument("--gpu", action="store_true", help="Docker GPU support")
    export.add_argument(
        "--dynamic-axes", action="store_true", default=True, help="ONNX dynamic axes"
    )

    dataset = sub.add_parser("dataset", help="Dataset management")
    dataset.add_argument("action", choices=["info", "validate", "split", "convert"])
    dataset.add_argument("--input", type=str, required=True)
    dataset.add_argument("--output", type=str)
    dataset.add_argument("--format", choices=["jsonl", "csv", "parquet"])
    dataset.add_argument("--split-ratio", type=float, default=0.8)

    augment = sub.add_parser("augment", help="Augment a dataset")
    augment.add_argument("input", type=str, help="Input dataset path")
    augment.add_argument("output", type=str, help="Output dataset path")
    augment.add_argument(
        "--techniques",
        nargs="+",
        default=["synonym", "swap"],
        choices=["synonym", "swap", "insert", "backtranslation"],
        help="Augmentation techniques",
    )

    analyze = sub.add_parser("analyze", help="Analyze dataset statistics")
    analyze.add_argument("dataset", type=str, help="Dataset path")

    registry = sub.add_parser("registry", help="Model registry management")
    registry_sub = registry.add_subparsers(dest="registry_action")
    registry_sub.add_parser("list", help="List registered models")
    registry_add = registry_sub.add_parser("add", help="Register a model")
    registry_add.add_argument("name", type=str, help="Model name")
    registry_add.add_argument("path", type=str, help="Model path")
    registry_add.add_argument("--metrics", type=str, help="Metrics JSON file")
    registry_add.add_argument("--tags", nargs="+", default=[], help="Tags")
    registry_compare = registry_sub.add_parser("compare", help="Compare two models")
    registry_compare.add_argument("model_a", type=str, help="First model name")
    registry_compare.add_argument("model_b", type=str, help="Second model name")
    registry_get = registry_sub.add_parser("get", help="Get model details")
    registry_get.add_argument("name", type=str, help="Model name")
    registry_delete = registry_sub.add_parser("delete", help="Remove a model")
    registry_delete.add_argument("name", type=str, help="Model name")

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
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_targets=args.lora_targets,
            scheduler=args.scheduler,
            warmup_steps=args.warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping=args.early_stopping,
        )
        return 0 if result else 1

    if args.command == "rag":
        from neuralforge.core.rag import RAGPipeline

        pipeline = RAGPipeline(display=display)
        if args.load_index:
            pipeline.load_index(args.load_index)
        if args.query:
            result = pipeline.query(args.query, top_k=args.top_k)
            display.print_rag_result(result)
        else:
            pipeline.build(args.docs, chunk_size=args.chunk_size, overlap=args.overlap)
        if args.save_index:
            pipeline.save_index(args.save_index)
        return 0

    if args.command == "evaluate":
        from neuralforge.core.evaluator import ModelEvaluator

        evaluator = ModelEvaluator(display=display)
        if args.compare_with:
            result = evaluator.compare_models(
                args.model, args.compare_with, args.dataset, args.metrics
            )
        else:
            result = evaluator.evaluate(args.model, args.dataset, args.metrics)
        if args.export_results:
            evaluator.export_results(result, args.export_results)
        display.print_eval_result(result)
        return 0

    if args.command == "export":
        from neuralforge.core.exporter import ModelExporter

        exporter = ModelExporter(display=display)
        kwargs = {}
        if args.format == "gguf":
            kwargs["quantization"] = args.quantization
        elif args.format == "onnx":
            kwargs["dynamic_axes"] = args.dynamic_axes
        elif args.format == "docker":
            kwargs["gpu"] = args.gpu
        result = exporter.export(args.model, args.format, args.output, **kwargs)
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

    if args.command == "augment":
        from neuralforge.core.dataset import DatasetManager

        manager = DatasetManager(display=display)
        manager.augment(args.input, args.output, args.techniques)
        return 0

    if args.command == "analyze":
        from neuralforge.core.dataset import DatasetManager

        manager = DatasetManager(display=display)
        manager.analyze(args.dataset)
        return 0

    if args.command == "registry":
        from neuralforge.core.registry import ModelRegistry

        registry = ModelRegistry(display=display)

        if args.registry_action == "list":
            models = registry.list_models()
            if not models:
                display.print_step("No models registered")
            else:
                display.print_step(f"Registered models ({len(models)}):")
                for m in models:
                    tags = ", ".join(m.get("tags", []))
                    display.print_step(f"  {m['name']} — {m.get('path', 'N/A')} [{tags}]")
        elif args.registry_action == "add":
            metrics = {}
            if args.metrics:
                import json

                with open(args.metrics) as f:
                    metrics = json.load(f)
            registry.register(
                {
                    "name": args.name,
                    "path": args.path,
                    "metrics": metrics,
                    "tags": args.tags,
                }
            )
        elif args.registry_action == "compare":
            registry.compare(args.model_a, args.model_b)
        elif args.registry_action == "get":
            model = registry.get_model(args.name)
            if model:
                display.print_step(json.dumps(model, indent=2, default=str))
            else:
                display.print_error(f"Model '{args.name}' not found")
        elif args.registry_action == "delete":
            registry.delete(args.name)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
