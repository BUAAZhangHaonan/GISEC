from __future__ import annotations

import sys

from gisec.train.train_gisec import parse_train_args, train_gisec


def main(argv: list[str] | None = None) -> None:
    train_gisec(parse_train_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main(sys.argv[1:])
