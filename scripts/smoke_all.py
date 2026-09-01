#!/usr/bin/env python3
"""Run one or more synthetic architecture smoke recipes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training_recipes.registry import architecture_catalog, task_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=task_names())
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    tasks = [args.task] if args.task else task_names()
    temporary = None
    output_dir = args.output_dir
    if output_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="recipe-smoke-")
        output_dir = temporary.name
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    completed = 0
    for task in tasks:
        for architecture in architecture_catalog(task):
            command = [
                sys.executable,
                "-m",
                "training_recipes.cli",
                "--task",
                task,
                "--architecture",
                architecture,
                "--profile",
                "smoke",
                "--max-steps",
                str(args.steps),
                "--output-dir",
                output_dir,
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
            completed += 1
    print(f"smoke suite complete: {completed} recipes")
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
