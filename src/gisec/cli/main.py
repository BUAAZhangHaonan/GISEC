"""Command line entrypoints for GISEC."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from gisec.train import (
    eval_gisec,
    infer_gisec,
    parse_eval_args,
    parse_train_args,
    train_gisec,
)

# One entry per subcommand: (help text, arg parser, handler). build_parser
# and main both derive their behavior from this table.
_COMMANDS: dict[str, tuple[str, Callable[[list[str]], Any], Callable[[Any], None]]] = {
    "train": ("Train a GISEC model", parse_train_args, train_gisec),
    "eval": ("Evaluate a trained GISEC model", parse_eval_args, eval_gisec),
    "infer": ("Run GISEC inference", parse_eval_args, infer_gisec),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gisec", description="Standalone GISEC command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _parse_args, _handler) in _COMMANDS.items():
        subparsers.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        build_parser().parse_args(args)
        return
    command, *remainder = args
    entry = _COMMANDS.get(command)
    if entry is None:
        build_parser().error(f"unknown command: {command}")
    _help_text, parse_args, handler = entry
    handler(parse_args(remainder))


if __name__ == "__main__":
    main(sys.argv[1:])
