# team_c — GPU (torch CUDA / cub radix) "值→名次" 排序

## 一句话

把"值→rank"整段搬上空闲的 RTX PRO 6000：pinned 异步 H2D →
`torch.sort(stable=True)`（cub 基数排序，1M f32 约 0.10 ms）→
`sv[1:] != sv[:-1]` 边界 + `cumsum`（int32 组号）→ `scatter_` 回原位置 →
D2H 回 numpy int32。CPU 侧只保留红线规定的 numba sobel/hypot
（直接 `from gisec.postproc_fast import _sobel_xy, _hypot_f32`，逐位不变）。

## 正确性论证（为什么逐位一致）

- **tie 不敏感**：rank 数组只依赖"值的划分"，不依赖 tie 组内顺序（同一组号
  scatter 给所有成员），所以 cub 基数排序（稳定）与 numpy mergesort 的
  组内顺序差异无关紧要——两个算法给出完全相同的值划分。
- **±0.0**：f32 键在排序前做 `k + 0.0` 归一（IEEE 下 `-0.0 + 0.0 = +0.0`，
  其余任何浮点 `x + 0.0 == x` 逐位不变）；即便不归一，组边界判据用的是
  IEEE `!=`（`-0.0 == +0.0`），两种机制都会让 ±0.0 共享 rank。实测
  fuzz `fz_negzero_tiny` 通过。（np.hypot 本身不可能产生 -0.0。）
- **mix 精确整数**：GPU 上按参考同样的 int64 算术
  `t[:n].to(int64) + 2 * t[n:].to(int64)`，先升 int64 再乘 2，无截断无溢出，
  与 numpy int64 逐位一致；i64 基数排序仅 0.13 ms，无需降位技巧。
- **输出**：rank 为 int32、dense 0..nrank-1 按值序，nrank 为 python int；
  `(H,W)` 任意形状（fuzz 的 257×509、2×3、1×1 均覆盖），空输入走守卫。
- **退化/环境稳健性**：torch 惰性 import（函数内），无 CUDA / `CUDA_VISIBLE_DEVICES=""`
  时自动退回参考的 numpy `_rank`（直接 import 复用，天然逐位一致）——
  CPU-only 机器 import 与 check 均不炸（已实测 check PASS）。

## 原文输出（最终跑）

```
$ cd <arena> && python harness.py check ../team_c/solution.py
CHECK PASS: bitwise identical on all real + fuzz cases

$ python harness.py bench ../team_c/solution.py
BENCH team_c: sem    16.8  mix     3.0  depth_cold    16.6  sem+mix    19.8  ms/img (mean of 20 imgs, median of 3 reps)

$ python harness.py refbench
BENCH reference(gisec.postproc_fast): sem   211.6  mix   163.0  depth_cold   188.1  sem+mix   374.6  ms/img
```

（CPU-only 回退路径同样 `CHECK PASS`；fork 两种模式见下节，均有实测。）

## 每函数毫秒数（两个口径）

单次调用延迟（bench 口径，20 图均值，**含 H2D/D2H 传输与同步**）：

| 函数 | team_c | 参考 | 加速 |
|---|---|---|---|
| rank_sem_logit | 16.8 ms | 211.6 ms | 12.6x |
| rank_mix | 3.0 ms | 163.0 ms | 54x |
| rank_depth_cold | 16.6 ms | 188.1 ms | 11.3x |
| sem+mix 合计 | 19.8 ms | 374.6 ms | 18.9x |

成分拆解（breakdown.py 实测，1M 元素 / 4MB f32）：

- **sem / cold**：CPU 红线段（sobel+hypot，numba 单线程）≈ 15.5 ms 是绝对
  大头；GPU 段（pinned H2D 4MB + sort + 边界/cumsum/scatter + D2H 4MB +
  同步）仅 **1.19 ms**，其中纯 GPU kernel 约 0.25 ms（f32 sort 0.10 +
  后处理 0.15），其余是传输与 launch/同步开销。hypot 直接写进 pinned
  buffer，省一次 4MB 中转 memcpy。
- **mix**：CPU 侧 2×4MB `np.copyto` 进 pinned ≈ 0.6 ms；GPU 段（8MB 单次
  H2D + int64 加法 + i64 基数排序 + 后处理 + D2H）≈ 2.4 ms。
- **不含传输口径**（数据已在显存、结果留在显存）：sort+后处理
  f32 ≈ 0.25 ms / i64 ≈ 0.28 ms——传输与同步占了 GPU 段的大头，PCIe
  pinned 带宽下 4MB 单向约 0.3-0.6 ms。

"连续处理 20 张图"摊销吞吐（trio = cold+sem+mix 顺序，含传输）：
**39.8 ms/img ≈ 25.1 img/s**（单函数摊销 sem 16.7 / mix 2.9 / cold 16.5 ms）；
参考同口径 trio ≈ 562.7 ms/img ≈ 1.78 img/s，约 **14x**。

诚实说明：sem/cold 已被红线 CPU sobel+hypot（~15.5 ms）封底，"值→rank"
本身（CPU 参考约 190-200 ms）已被压到 ~1.2 ms（含传输），再快只能动红线
梯度核，本队不做。mix 无 CPU 红线段，3.0 ms 里大头是两次 4MB 上行与
一次 4MB 下行。

## fork 集成问答（fullval：fork 出 16 个 CPU worker，父进程 fork 之后才初始化 CUDA）

**(a) worker 内各自初始化 CUDA 是否可行？——可行，且已实测，是推荐方案。**

关键事实：CUDA context 不能跨 fork 存活，但 fullval 的顺序恰好是安全的——
`fullval.py` 第 193 行 `mp.get_context("fork").Pool(16)` 在第 200 行
`model.cuda()` **之前**，fork 时父进程还没碰过 CUDA，因此每个 worker 是
"CUDA 干净"进程，可以各自惰性建自己的 context（标准多进程 CUDA 用法）。

实测（test_fork.py）：
- 模式 A（先 fork 后用 GPU，即 fullval 模式）：3 个子进程各自建 context，
  全部走 GPU 路径，输出与参考逐位一致。
- 模式 B（父进程先用 GPU 再 fork）：CUDA 硬限制，子进程用 GPU 会
  `cudaErrorInitializationError` 直接 abort（连 Python except 都接不住，
  异常展开时 tensor 析构再次触发 C++ terminate）。本方案对此做了防御：
  `os.register_at_fork(after_in_child=...)` 把子进程 `_BACKEND` 置为
  False（CPU 回退），并**故意泄漏**继承的 pinned/CUDA 句柄使其析构函数
  永不运行——实测子进程 exitcode 0、输出仍逐位一致（退回参考 CPU 速度，
  不炸不挂）。也就是说：即便未来有人把顺序改坏，最坏结果是退回 212ms
  基线，而不是崩溃。

代价核算（16 worker）：
- 显存：实测每进程 714 MiB（context + 缓存分配器；pinned buffer 在主机
  内存，每 worker 约 12 MiB）→ 16 × 0.72 ≈ **11.5 GiB / 98 GiB**，宽裕。
- 初始化：每 worker 一次性 ~0.15-0.5 s（惰性，首次 rank 时），16 张/worker
  摊销可忽略；harness warm 两次已覆盖同进程口径。
- 争用：每图 GPU 真实占用 ≈ 3.6 ms（sem 1.2 + cold 1.2 + mix 2.4 里的大
  部分），纯 kernel 仅 ~0.5 ms；16 worker 时间片轮转即使全串行化 GPU 段，
  摊销 ~3.6 ms/img，仍低于新的瓶颈（CPU sobel 15.5 ms，它天然 16 进程并
  行）。GPU 不会成为新瓶颈，单次延迟可能抖动几 ms；如在意可开 CUDA MPS
  （`nvidia-cuda-mps-control`）减少 context 切换并允许 kernel 重叠——
  不开也能用（默认时间片）。

**(b) 若不想让 worker 碰 GPU：主进程算好 rank 再下发。**

替代接线（改动最小者优先）：
1. **父进程预计算 + process() 加透传参数**：`_worker_one` 的 payload
   `(meta, sem_logit, hm, off, depth)` 在父进程推理循环里本来就逐图生成
   （sem_logit 来自父进程 GPU 前向，depth 来自磁盘），在 payload 里追加
   父进程算好的 `(rank, nrank)`（父进程 GPU：cold+sem+mix ≈ 36.8 ms/图，
   其中 31 ms 是父进程单线程 sobel/hypot——会把父进程循环变成新瓶颈，
   需与前向耗时权衡）；`postproc_fast.process` 加可选参数
   `rank=None, nrank=None`，非 None 时跳过 `load_or_compute_rank` /
   `sem_logit_rank` / `mix_elevation_rank` 三步。改动约 10 行，算法零变化。
2. **独立 rank-server 进程**：一个专职进程独占 GPU 排序，16 个 CPU worker
   把 sem/depth（4MB）经共享内存环或 pipe 发给它，回传 int32 rank。worker
   完全不碰 CUDA，父进程也不用串行做 sobel；代价是自建 RPC。
3. **零接线兜底**：depth rank 本就有磁盘缓存链路
   （`python -m gisec.postproc_fast` 预计算，`load_or_compute_rank` 命中
   `np.load`），生产中冷路径只剩 sem+mix——保持 (a) 即可，无需 (b)。

**原地替换成本**：三个函数签名/返回与参考一一对应，可直接
`postproc_fast.sem_logit_rank = team_c.rank_sem_logit`（或改三处函数体为
委托调用）；numba `cache=True` 内核不受影响；无 GPU 环境自动落回参考实现，
依赖干净（torch 仅惰性 import，且 import solution.py 时不触发）。

## 环境纪律

- 所有测试均为短时运行（单条 < 2 min），未用 systemd-run；GPU 期间确认
  空闲（bench 时无他人进程）。
- 只写了 team_c 目录；`src/gisec` 与 arena 未改动。
- 附带脚本：`explore.py`（分项计时探索）、`test_fork.py`（fork 两种模式
  实测）、`breakdown.py`（NOTES 计时口径复现）。
