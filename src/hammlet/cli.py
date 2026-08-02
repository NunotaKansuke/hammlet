"""Command-line entry point for schedulers and multi-machine atlas builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_atlas, merge_parts
from .config import AtlasConfig, ParameterGrid, Partition


def _config(path: Path):
    content = json.loads(path.read_text(encoding="utf-8"))
    grid_data = content["grid"]
    if {"log_s", "log_q", "log_rho"}.issubset(grid_data):
        grid = ParameterGrid.from_log10(**grid_data)
    else:
        grid = ParameterGrid(**grid_data)
    return grid, AtlasConfig(**content.get("atlas", {}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hammlet")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build one atlas or one partition")
    build.add_argument("config", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--part-index", type=int, default=0)
    build.add_argument("--part-count", type=int, default=1)
    build.add_argument("--progress-every", type=int, default=1)
    merge = commands.add_parser("merge", help="merge a complete set of parts")
    merge.add_argument("output", type=Path)
    merge.add_argument("--destination", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        grid, config = _config(args.config)
        result = build_atlas(
            args.output,
            grid,
            config=config,
            partition=Partition(args.part_index, args.part_count),
            progress_every=args.progress_every,
        )
    else:
        result = merge_parts(args.output, destination=args.destination)
    print(result)
    return 0

