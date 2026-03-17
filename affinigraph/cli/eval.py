from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    from affinigraph.train.train_reference_unet_gnn import eval_main, parse_eval_args

    eval_main(parse_eval_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
