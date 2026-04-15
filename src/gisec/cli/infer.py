from __future__ import annotations

import sys

from gisec.train.train_gisec import infer_gisec, parse_infer_args


def main(argv: list[str] | None = None) -> None:
    infer_gisec(parse_infer_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main(sys.argv[1:])
