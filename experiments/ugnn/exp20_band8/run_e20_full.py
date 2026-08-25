"""Run the full 3276 fast FINAL for E20 at its sweep-winning SEM_THR 0.9
without touching the eval_centernet default (fork-time module global)."""
import sys

sys.path.insert(0, "/home/k100/zhn/electronic-components-grasp-and-segment/gisec/experiments/ugnn/exp09_centernet_seeds")
sys.path.insert(0, "/home/k100/zhn/electronic-components-grasp-and-segment/gisec/experiments/ugnn/exp08_scale_32254")
sys.path.insert(0, "/home/k100/zhn/electronic-components-grasp-and-segment/gisec/experiments/ugnn/exp03_unet_dense")
import eval_centernet as ec

ec.SEM_THR = 0.9
HERE = "/home/k100/zhn/electronic-components-grasp-and-segment/gisec/experiments/ugnn/exp20_band8"
sys.argv = [
    "eval_centernet.py",
    "--arch", "e10",
    "--profile", "fast",
    "--ckpt", f"{HERE}/runs/best.pth",
    "--out", f"{HERE}/eval_full_fast_090.json",
]
ec.main()
