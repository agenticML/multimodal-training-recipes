"""Command-line entrypoints."""

from __future__ import annotations

import argparse
import json
import os

from .config import load_document, resolve_config
from .models import UnsupportedDummyArchitecture
from .registry import architecture_catalog, architecture_spec, recipe_path, task_names
from .trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multimodal Hugging Face training recipe")
    parser.add_argument("--task", choices=task_names(), required=True)
    parser.add_argument("--architecture")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--dataset-path")
    parser.add_argument("--loss")
    parser.add_argument("--optimizer")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--resume-from")
    parser.add_argument("--output-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--list", action="store_true", help="List architectures for the selected task"
    )
    return parser


def apply_cli_overrides(args: argparse.Namespace, config: dict) -> dict:
    if args.dataset_path is not None:
        config["data"]["dataset_path"] = args.dataset_path
    if args.loss is not None:
        spec_losses = config.get("_compatible_losses")
        if spec_losses is not None and args.loss not in spec_losses:
            raise ValueError(f"loss {args.loss!r} is not compatible; expected one of {spec_losses}")
        config["loss"] = [{"name": args.loss, "weight": 1.0}]
    if args.optimizer is not None:
        config["optimizer"]["name"] = args.optimizer
    if args.learning_rate is not None:
        config["optimizer"]["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.precision is not None:
        config["training"]["precision"] = args.precision
    if args.resume_from is not None:
        config["training"]["resume_from"] = args.resume_from
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    config.pop("_compatible_losses", None)
    return config


def _build_config(args: argparse.Namespace) -> dict:
    spec = architecture_spec(args.task, args.architecture)
    recipe = load_document(recipe_path(args.task, args.architecture))
    config = resolve_config(recipe, args.profile)
    config["source"] = spec["source"]
    config["task_kind"] = spec.get("task_kind")
    config["family"] = spec.get("family")
    config["evaluation"] = spec["evaluation"]
    config["_compatible_losses"] = spec["losses"]
    apply_cli_overrides(args, config)
    if args.profile == "full" and not config["data"].get("dataset_path"):
        raise ValueError("the full profile requires --dataset-path")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return config


def run(args: argparse.Namespace) -> None:
    if args.list:
        for name, spec in architecture_catalog(args.task).items():
            print(f"{name:28} {spec['family']:24} {spec['source']['hub_id']}")
        return
    if not args.architecture:
        raise ValueError("--architecture is required unless --list is used")
    try:
        print(json.dumps(train(_build_config(args)), indent=2, sort_keys=True))
    except UnsupportedDummyArchitecture as exc:
        print(f"skip {args.task}/{args.architecture}: {exc}")


def main(argv: list[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
