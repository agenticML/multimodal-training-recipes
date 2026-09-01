#!/usr/bin/env python3
"""Check pinned Hugging Face revisions against current repository heads."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training_recipes.registry import architecture_catalog, task_names


def main() -> None:
    drift: list[dict[str, str]] = []
    checked: set[str] = set()
    for task in task_names():
        for spec in architecture_catalog(task).values():
            source = spec["source"]
            hub_id = source["hub_id"]
            if hub_id in checked:
                continue
            checked.add(hub_id)
            with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{hub_id}", timeout=30
            ) as response:
                current = json.load(response)["sha"]
            if current != source["revision"]:
                drift.append({"hub_id": hub_id, "pinned": source["revision"], "current": current})
    print(json.dumps({"checked": len(checked), "drift": drift}, indent=2, sort_keys=True))
    if drift:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
