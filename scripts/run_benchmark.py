#!/usr/bin/env python3
"""CLI entry point for foundation model adaptation benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from fm_adapt.evaluation.runner import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FM adaptation benchmarks")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/adaptation_benchmark.yaml"),
        help="Benchmark config YAML",
    )
    parser.add_argument(
        "--module",
        choices=["adaptation", "eval", "attribution", "all"],
        default="adaptation",
        help="Which benchmark module to run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for benchmark outputs",
    )
    args = parser.parse_args()
    run_dir = run_benchmark(args.config, module=args.module, output_dir=args.output_dir)
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
