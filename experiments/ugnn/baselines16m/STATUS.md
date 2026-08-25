# baselines16m: parameter-matched classic detectors on 32254

Goal: "GISEC vs classic detectors at the same parameter count and the
same budget". Three baselines in the 14-18M band (GISEC E10 reference:
16.85M), trained on datasets/20260318_1K_32254 (train 25654 / val 3276),
20 epochs, batch 8 @ 1024 direct read, 3206 steps/epoch, bf16 AMP.

## Implementation route
- detectron2 is NOT installed in the gisec env; src/gisec's Mask2Former
  is the HuggingFace `transformers` implementation, and the old 47M
  baseline training scripts were removed in the repo refactor (only
  output artifacts survive). Therefore these baselines are built on
  torchvision (Mask R-CNN) and HF transformers (Mask2Former with a
  timm resnet18 backbone), reusing gisec.datasets.coco_utils for data.
- ImageNet init: torchvision resnet18 (cached) for MRCNN, timm
  resnet18 (downloaded once via hf-mirror, then HF_HUB_OFFLINE=1).

## The three configs
| family | model | width/depth | LR (community standard) | params |
|---|---|---|---|---|
| mrcnn16 | Mask R-CNN R18-FPN(256), FastRCNNConvFCHead (192,192), mask head (64,64) | 5 anchors/level, 1024 min/max size | SGD 0.02 mom 0.9 wd 1e-4 | 17.0M |
| m2f16 | HF Mask2Former, timm R18 (out_indices 1-4) | feature/hidden 160, pixel-decoder 4 layers, transformer-decoder 10 layers, FFN 640, 100 queries | AdamW 5e-5 (backbone 0.1x) wd 0.05 | 16.54M |
| m2f16cat | m2f16 + 4ch stem (depth channel = RGB-mean init of conv1) | depth calibrated (d-0.245)/(0.686-0.245) clamp[-1,2] | same as m2f16 | 16.54M |

All three: 500-step linear warmup + cosine decay, workers 8
prefetch 4 (instance masks packed with np.packbits to keep worker IPC
at ~1/8 of raw mask bytes), 1 class (component, category id 1).

## Smoke anchors (50 steps, appended by run_one.sh)

(placeholder — filled by the chain)

## Chain / lock protocol
- chain.sh launches run_one.sh per family in systemd user unit
  `baselines16m-<family>` (MemoryMax=160G, CPUQuota=3200%), in order
  mrcnn16 -> m2f16 -> m2f16cat; a stage failure stops the chain.
- Each stage: 50-step smoke first (s/step + peak VRAM appended here),
  then full 20-epoch training, then eval (COCO segm AP on val 3276,
  score 0.05 / mask 0.5) appended to RESULT.md.
- GPU priority lock: while /tmp/gisec_gpu_priority exists, the chain
  sleeps 60 s before launching a stage, and before starting the full
  training inside a stage. The GISEC mainline experiments can create
  that file to preempt the chain between trainings.

## Results

See RESULT.md (appended after each eval).
- mrcnn16 smoke: params 17.00M, 0.22 s/step, peak 6.8 GiB, ETA 3.8 h for 20 epochs

## Incident 2026-08-24: mrcnn16 eval OOM
- 根因: eval.py predict_* 把全部 3276 张图的全分辨率 uint8 掩码 (1024x1024x实例数)
  累积在 results 列表里直到 COCOeval 前才 RLE 化, 与 E17 评测叠加内存后触发 cgroup OOM
  (11:09, unit result=oom-kill)。训练产物 model_final.pth 完好。
- 修法: 逐图即时 mask_util.encode RLE 化 (emit 回调内联进推理循环), 掩码用完即 del,
  json_results 只存 RLE 级 COCO dict, RAM 全程平稳 (~12 GiB)。
- 重跑: baselines16m-mrcnn16-eval (MemoryMax=64G) 全量 3276 图, 1345 s 完成,
  segm AP 0.6082 / AP50 0.8649 / AP75 0.6926 (bbox AP 0.6840)。已写入 RESULT.md。
- 链重启: mrcnn16 跳过 (已训完), chain_rest.sh (m2f16 -> m2f16cat) 于 13:35 启动。
- m2f16 smoke: params 16.54M, 0.64 s/step, peak 25.4 GiB, ETA 11.5 h for 20 epochs
- m2f16cat smoke: params 16.54M, 0.67 s/step, peak 25.5 GiB, ETA 12.0 h for 20 epochs
- m2f16 训完并评测: segm AP 0.43393 / AP50 0.62843 / AP75 0.52561 / APs≈0 / APm 0.45592, bbox 0.37456 (详见 RESULT.md)。m2f16cat 已由链自动启动 (systemctl is-active = active, 08-25 09:54 起)。
- m2f16cat 训完并评测: segm AP 0.22437 / AP50 0.39306 / AP75 0.24364 / APs 0.0 / APm 0.26398, bbox 0.04918 (详见 RESULT.md)。
- 收官: 基线链 3/3 完成 (mrcnn16 -> m2f16 -> m2f16cat), 2026-08-25。总结表见 RESULT.md。
