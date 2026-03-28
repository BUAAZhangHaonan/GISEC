from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    from gisec.train.train_gisec import parse_train_args, train_main

    train_main(parse_train_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
