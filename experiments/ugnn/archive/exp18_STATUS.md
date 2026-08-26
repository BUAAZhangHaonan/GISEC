# E18 depth-only input

- 状态: DONE 2026-08-25, PARTIAL (线①过 线② FAIL, 不进 canonical; 见 RESULT.md)
- 代码: train_depth_only.py (fork exp10 train_capacity.py)
- 单变量 vs E10: 输入 4ch -> 1ch (仅深度, 标定与 4ch 深度通道逐位同值); conv1 用 ImageNet 3ch 核按通道均值初始化; 其余全同 E10 (含 SEM_W=2, AdamW 3e-4 cosine 20ep, 偶数 epoch val, best 按 mIoU)。无 E17 的 band/EMA。
- 动机: 6401 d2m2f depth_only 78.94 > concat 76.29; GISEC 单类前景分割, E1 取证外观 AUC 0.584 ≈ 随机。
- 预注册: RESULT.md
- 产物: runs/best.pth, runs/last.pth, runs/train_log.json
