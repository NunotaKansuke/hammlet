"""Command-line entry point for schedulers and multi-machine map builds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build_maps, merge_maps
from .config import MapConfig, ParameterGrid, Partition


def _config(path: Path):
    content = json.loads(path.read_text(encoding="utf-8"))
    grid_data = content["grid"]
    if {"log_s", "log_q", "log_rho"}.issubset(grid_data):
        grid = ParameterGrid.from_log10(**grid_data)
    else:
        grid = ParameterGrid(**grid_data)
    return grid, MapConfig(**content.get("maps", {}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hammlet")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-maps", help="build maps or one partition")
    build.add_argument("config", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--part-index", type=int, default=0)
    build.add_argument("--part-count", type=int, default=1)
    build.add_argument("--progress-every", type=int, default=1)
    merge = commands.add_parser("merge-maps", help="merge a complete set of parts")
    merge.add_argument("output", type=Path)
    merge.add_argument("--destination", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-maps":
        grid, config = _config(args.config)
        result = build_maps(
            args.output,
            grid,
            config=config,
            partition=Partition(args.part_index, args.part_count),
            progress_every=args.progress_every,
        )
    else:
        result = merge_maps(args.output, destination=args.destination)
    print(result)
    return 0
