# C 档极限竞技场 — 规则（2026-09-05）

目标：把 GISEC 单图推理压到极限。探索性质，**不进默认链**；接受精度
漂移（fp16/TensorRT/非逐位 kernel 都可以），但质量用下游 AP 度量并
如实上报。硬约束：**目标硬件 RTX 3090 24GB**——所有代码必须能在
24 GiB 显存预算内跑（k100 上用 `torch.cuda.set_per_process_memory_fraction(
24.0/97.9)` 强制；harness 已带，你的 runner 也要带）；只用本机 GPU 0。

## 基线（k100 实测，E26b ckpt @0.95，见 baseline.json）
- 前向（torch eager fp32, bs1）: ~14 ms
- CPU 值→rank（已优化的 radix）: ~77 ms
- numba watershed+merge: ~46 ms
- gpu_fast 全链（GPU 段 21 + CPU watershed 60 重叠）: wall ~72 ms/图

## 轨道与接口（solution.py 按轨道实现对应函数）

### ws 轨道（GPU watershed，最大瓶颈，优先级最高）
```python
def ws_labels(rank, nrank, sem, markers) -> np.int32 (H, W) labels
```
- 语义参考 `gisec.postproc_fast._ws_bucket`：优先级洪泛（bucket 队列、
  值钳位 child=max(child,parent)、marker 即标签、sem==0 处 label 0）；
  merge(<32px) 由 harness 的 CPU tail 做，你只管 labels。
- 允许改变 tie 顺序/并行策略（如 level-set 分桶并行、sorted-order
  union-find、迭代标签传播），质量门 = 下游 AP delta > -0.005（40 图
  子集）——Ampere 显卡上 triton/load_inline CUDA/numba-cuda 均可。
- rank 是 int32 稠密海拔（值域 [0, nrank)，nrank≈1M），sem u8，markers
  int32（marker k 在第 k 个坐标处，k=1..K）。

### fwd 轨道（前向替换）
```python
def fwd(img_u8, depth_f32) -> (sem_logit, hm, off)
```
- 语义 = `gisec.inference._forward`（同输入预处理；hm=sigmoid(seed[0,0]),
  off=seed[0,1:3]——注意 harness 直接拿你的 hm/off 走 CPU 解码）。
- 武器任选：TensorRT（已装 11.2 + onnx；fp16 首选，fp32 附带一份做逐位
  对照）、torch.compile(max-autotune)+CUDA graphs、bf16 autocast。
- 质量门 = 下游 AP delta > -0.005（40 图）；engine 构建（一次性）排除
  在计时外，推理延迟含 H2D/D2H。

## 环境纪律（红线）
- 重负载必须 systemd-run --user --unit=<名> -p MemoryMax=24G --wait --
  /home/k100/miniconda3/envs/gisec/bin/python ...；显存预算强制如上。
- 只写自己的 team 目录；不改 src/gisec、不改 arena。
- python: /home/k100/miniconda3/envs/gisec/bin/python（gisec 已 -e 安装；
  tensorrt/onnx/triton/nvcc 可用）。
- 自测命令：
  cd <arena>; python harness.py ws ../team_X/solution.py   # 或 fwd
- 交付：solution.py + NOTES.md（思路/结果原文/显存峰值/3090 适配注记）。
  报数字必须来自真实运行，禁止估算冒充实测。
