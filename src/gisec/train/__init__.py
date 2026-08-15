"""Training and evaluation entrypoints."""

from gisec.train.args import parse_eval_args, parse_infer_args, parse_train_args
from gisec.train.evaluate import eval_gisec, infer_gisec
from gisec.train.trainer import train_gisec

__all__ = [
    "eval_gisec",
    "infer_gisec",
    "parse_eval_args",
    "parse_infer_args",
    "parse_train_args",
    "train_gisec",
]
