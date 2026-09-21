#!/usr/bin/env python3
"""Combine one event's light curve and Δχ² map into one image."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", type=Path, help="one completed event result directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG (default: EVENT/figures/event-summary.png)",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    event = args.event.expanduser().resolve()
    lightcurve = event / "figures" / "lightcurve.png"
    chi2_map = event / "figures" / "chi2_map.png"
    for path in (lightcurve, chi2_map):
        if not path.is_file():
            raise SystemExit(f"missing plot: {path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.4, 5.4),
        squeeze=False,
        constrained_layout=False,
    )
    axes[0, 0].imshow(mpimg.imread(lightcurve), aspect="auto")
    axes[0, 1].imshow(mpimg.imread(chi2_map), aspect="auto")
    for axis in axes[0]:
        axis.axis("off")
    figure.suptitle(f"Roman local event {event.name}", fontsize=15)
    figure.subplots_adjust(
        left=0.002,
        right=0.998,
        bottom=0.002,
        top=0.92,
        wspace=0.015,
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else event / "figures" / "event-summary.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi)
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
