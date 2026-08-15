from __future__ import annotations

import sys

from gisec.train import eval_gisec, parse_eval_args


def main(argv: list[str] | None = None) -> None:
    eval_gisec(parse_eval_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main(sys.argv[1:])
