"""python -m training_recipes.dashboard"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Localhost smoke-run dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--asset-dir", required=True, help="AgenticML offline asset directory")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    asset_dir = Path(args.asset_dir).expanduser()
    asset_dir.mkdir(parents=True, exist_ok=True)
    import uvicorn

    from .app import create_app

    app = create_app(asset_dir=asset_dir, host=args.host)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
