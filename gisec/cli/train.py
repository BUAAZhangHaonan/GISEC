from __future__ import annotations

import sys

from gisec.cli._routing import should_route_legacy


def main(argv: list[str] | None = None) -> None:
    from gisec.cli.train_legacy import main as legacy_main
    from gisec.train.train_active import parse_train_args, train_active

    argv = list(sys.argv[1:] if argv is None else argv)
    if should_route_legacy(argv):
        legacy_main(argv)
        return

    train_active(parse_train_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
