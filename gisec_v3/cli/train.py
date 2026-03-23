from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gisec_v3.cli.train",
        description="GISEC v3-alpha query-only object-first training surface.",
    )
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-family", default="UQ")
    parser.add_argument("--model-scale", choices=("s", "m"), default="s")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
