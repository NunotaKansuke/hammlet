"""CLI for creating a read-only local packed bucket cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from .packed import pack_atlas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--m-max",
        type=int,
        default=None,
        help="stored mode limit (default: the source atlas limit, normally 512)",
    )
    parser.add_argument("--max-maps", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse matching bucket files already present in the destination",
    )
    args = parser.parse_args(argv)
    pack_atlas(
        args.atlas,
        args.output,
        m_max=args.m_max,
        max_maps=args.max_maps,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
