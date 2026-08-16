"""Command line entrypoints for GISEC."""
from __future__ import annotations

import argparse
import sys

from gisec.train import eval_gisec, infer_gisec, parse_eval_args, parse_train_args, train_gisec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gisec", description="Standalone GISEC command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train a GISEC model")
    subparsers.add_parser("eval", help="Evaluate a trained GISEC model")
    subparsers.add_parser("infer", help="Run GISEC inference")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        build_parser().parse_args(args)
        return
    command = args[0]
    remainder = args[1:]
    if command == "train":
        train_gisec(parse_train_args(remainder))
        return
    if command == "eval":
        eval_gisec(parse_eval_args(remainder))
        return
    if command == "infer":
        infer_gisec(parse_eval_args(remainder))
        return
    build_parser().error(f"unknown command: {command}")


if __name__ == "__main__":
    main(sys.argv[1:])
