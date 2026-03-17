from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    from affinigraph.train.train_reference_unet_gnn import infer_main, parse_infer_args

    infer_main(parse_infer_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
