from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    from gisec.train.train_gisec import eval_main, parse_eval_args

    eval_main(parse_eval_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
