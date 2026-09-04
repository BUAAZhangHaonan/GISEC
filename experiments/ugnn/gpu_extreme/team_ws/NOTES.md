# team_ws — GPU watershed (ws track)

## 结果（真实实测，k100 / RTX PRO 6000 Blackwell, CUDA 13.1, torch 2.10）

官方自测（systemd-run -p MemoryMax=24G）：

```
WS-RESULT team_ws: labels 10.00 ms/img (numba ws+merge 46.4)
| AP 0.93824 vs canonical 0.93847 (delta -0.00023) | peak VRAM 0.06 GiB
```

（直跑两遍复测：10.06 / 10.16 ms，AP delta 均为 -0.00023，数字稳定。）

- 质量门（AP delta > -0.005）：通过，余量 0.0048。
- 比 numba `_ws_bucket`（43.5-46.4 ms/图）快 **~4.5x**；全链视角：CPU ws 段
  60ms → ~10ms，gpu_fast 全链 72ms → 预计 ~25ms 量级（未实测全链）。
- 逐像素标签 vs numba 参考：不匹配 0.003%-0.03%（全为 tie 顺序差与量化
  边界），**无孤儿像素**（labels==0 且有已标记 4-邻的 sem 像素 = 0），
  无标签丢失（harness tail 只扫 marker 标签，全部存在）。

## 算法（一句话）

rank 量化到 K=16 个 level 桶，按桶升序做 minimax 洪泛：每桶先做
"从未标记邻域取 min 标签"的 seed 扫描，再用 frontier BFS 波 + 投机 2-hop
claim 并行扩散；claim 门 `q[n] <= b`（像素可在自己桶过后被更高层级洪泛
补占）在桶粒度上复现了 numba 的值钳位（child=max(child,parent)）语义。

## 实现

单文件 `solution.py`，torch `load_inline` 编译两个 kernel（build/ 缓存，首次
编译 ~40s，之后即时加载）：

1. `ws_pre_kernel`（256 blocks × 512 thr）：labels/meta 初始化 + 每 block
   直方图 partial。meta 打包 `q | sem<<16 | marker<<17`（每邻居 1 次 load）。
2. `ws_sweep_kernel`（**cooperative launch, 2 blocks × 1024 thr**，
   `__launch_bounds__(1024,1)` 限 64 reg）：直方图归约+前缀 → counting-sort
   scatter（全局原子 cursor）→ 桶循环 {seed 扫描 → 波循环}。
   - frontier 表项打包 `(label<<20 | pixel)` u32，pop 免 label 读。
   - 每 wave 一次 `grid.sync()`；8 组轮转 frontier buffer，wave W 清零
     cell (W+4)&7（上次读在 W-4，下次写在 W+3，无清零竞态）。
   - 波内 claim：13 邻域 meta 一批并行预取 → 4 个 hop-1 CAS 并行 →
     hop-2 CAS 仅当所挂靠的 hop-1 像素 meta 可通行（sem && !marker && q<=b）。
   - 桶切换处（读到 c==0 后）**必须加 barrier** 再 seed 下一桶（见踩坑）。

Host 侧：pinned 中转 + 异步 H2D/D2H，所有 GPU buffer 跨调用复用
（`_Ctx`），返回时 pinned→新 numpy 数组（安全拷贝 0.45ms）。
计时含 H2D/D2H（harness 口径）。

## K 扫描（40 图子集实际跑的是 K=16；5 图 CPU numba 模拟器扫的 K 敏感性）

| K | waves(2-hop, GPU) | 标签不匹配 vs numba | harness AP delta |
|---|---|---|---|
| 8 | ~470 | 0.025% | **-0.00342**（过门但余量小） |
| 16 | ~832 | 0.013% | **-0.00023**（选定） |
| 32 | ~1276 | 0.008% | （未跑；更快档已够） |
| 64 | ~2072 | 0.009% | -0.00023（早期单 block 版本） |

- 不匹配率在 K=16→64 已饱和（残差是 tie 顺序，非量化误差）。
- K=8 的 AP 掉得多（-0.0034）不是不匹配像素变多，而是个别实例边界翻转
  触发 merge/box 变化；为门限余量选 K=16。
- NBLK 扫描（K=16）：1/2/3/4/6/8 blocks kernel 8.12/7.61/7.79/7.92/8.75/9.63 ms
  —— grid.sync 随 block 数变贵，2 blocks 最优；≥8 反而劣化。

## 迭代次数 / 残差

- 外层桶数 K=16；波数（BFS 深度，2-hop）~830/图（dbg[0] 上报，非迭代上限）。
- wave_cap=4,000,000 兜底（从未触发；实际 ~830）。
- 收敛即终止（frontier 空），无固定迭代次数、无残差像素
  （sem 内无 marker 的孤岛与 numba 一样保持 0，语义一致）。

## 显存峰值

0.06 GiB（harness 上报，含 torch 上下文）。工作集：rank/labels/meta/lst/
bufs(8×4MB)/pinned ≈ 60MB，远低于 24GB 预算；`set_per_process_memory_fraction
(24/97.9)` 在模块导入时已设。

## 3090 适配注记

- kernel 经 `__launch_bounds__(1024,1)` 限 64 寄存器 → 每 SM 1 个
  1024-thread block；cooperative launch 用 2 blocks，3090（82 SM）必然
  co-resident，无死锁风险。若增大 NBLK 需 ≤ SM 数。
- 编译时需 `TORCH_CUDA_ARCH_LIST="8.6"`（本机默认取当前设备 12.0；扩展
  构建缓存在 team_ws/build，换卡删掉重建即可）。代码仅用 int32/u32 原子与
  grid.sync，无 fp8/新指令依赖。
- 共享内存 (K+2)*4B ≈ 72B/block，忽略不计。

## 踩坑记录（多 block 屏障，血泪）

1. **桶切换竞态（根因）**：波循环读到 c==0 后直接进入下一桶 seed，
   快 block 的 seed push 与慢 block 的读在同一 cell 上竞争 → 各 block
   波数分歧 → grid.sync 配对不同阶段 → 随机丢 frontier（孤儿像素）或
   死锁。修复：c==0 判定后加一次 grid.sync 再 seed。
2. cells 轮转计数器跨调用残留 → kernel 开头 grid-stride 清零全部 8 个。
3. **grid.sync() 不可放在运行时条件分支里**（`if(flag) ...else grid.sync()`）：
   编译可过但 ≥2 blocks 死锁（CG 要求全线程无条件到达）。手写原子 barrier
   亦实测更慢（64 blocks ~15-25us/次），放弃，统一无条件 grid.sync。
4. cudaLaunchCooperativeKernel 会做 co-residency 检查，超限报
   "too many blocks"（曾因加代码寄存器涨到 >64 触发）。
5. 计数器读用 `atomicAdd(cell,0)` 走 L2，避免跨 SM 的 L1 陈旧值
   （防御性；最终根因见 1）。

## 已知余量（未做完的进一步提速方向）

- 每 wave 成本 ~7us（grid.sync + 3 级访存链）× 830 波 ≈ 5.8ms 是主导；
  frontier 进 shared memory（3090 限 99KB ≈ 25k 表项，需溢出路径）或
  frontier 表项附带邻居 meta（省一批 meta 读）可再砍一半，工程量中等。
- Python 侧固定开销 ~2ms（pinned 拷贝 0.14 + H2D 0.45 + D2H 0.31 +
  返回拷贝 0.45 + 胶水）：双流重叠 H2D、返回视图化可再省 ~0.7ms（后者
  牺牲接口安全性，未做）。

---

# R3 更新（ws_labels 提速 + ws_full）

## 结果（systemd-run MemoryMax=24G 官方口径，K=10 默认）

```
WS-RESULT team_ws: labels 6.82 ms/img (numba ws+merge 46.4)
| AP 0.93715 vs canonical 0.93847 (delta -0.00132) | peak VRAM 0.06 GiB
SELFCHECK ws_full: bitwise merge/boxes vs numba: PASS (40/40 图)
WSFULL ms/img: 7.04   （含 watershed+merge+boxes+全部 H2D/D2H）
```

- **ws_labels：10.18 → 6.60-6.82 ms/图**（目标 ≤7 达成；vs numba 46.4 = **6.8x**）。
  直跑复测 6.60（K=10）systemd-run 6.82。K=8 可到 6.38-6.41（AP -0.0034，余量偏小）。
- **ws_full（新增）**：7.04 ms/图完成 洪泛+merge+boxes，逐位复现 numba 语义
  （_selfcheck.py：固定输入回灌 GPU tail，40/40 图 labels/boxes 逐位一致）；
  下游 AP 与 ws_labels 相同（-0.00132），消灭原 CPU 尾 ~8ms（_merge ~4ms + _boxes ~4ms）。
- 返回 labels 为 4 个轮转 pinned 视图之一（有效期 = 之后 4 次 ws 调用之内；
  WS_SAFEOUT=1 恢复安全拷贝，代价 +0.45ms）。

## R3 提速手段（按贡献排序）

1. **投机 2-hop claim 反而是负优化**：每 pop 13 次 meta 预取 + 12 次 CAS 的
   成本 > 波数减半的收益。关掉（hop-1 only）后 waves 829→1628 但 kernel
   7.61→6.47ms。教训：波数 × 每波成本要乘出来比。
2. **seeds 只 claim 不 push**：桶 seed 阶段不再把 ~28 万种子压进 frontier
   （原是单地址全局原子大头），wave-0 直接扫桶列表取已标记像素作源。
3. **warp 聚合 push**（__activemask 模式，分叉安全）：每 active warp 组每轮
   1 次 atomicAdd，lance 稠密写入，无空隙。配合 2 后全局原子从 ~47 万次
   降到 ~1.5 万次。
4. **K 重扫（hop-1）**：8/10/12/16 → 6.38/6.60/7.23/8.3 ms，
   AP -0.0034/-0.0013/-0.0023/~-0.0002。K=10 为速度/质量平衡点（默认）。
5. **返回视图轮转** + 去掉 mkf.max() 全量断言：-0.75ms Python 侧。

## R3 排除的路线（实测失败，勿踩）

- shared-memory staging frontier（写共享、每波 flush 到全局）：sstat 动态
  共享未初始化导致越界（compute-sanitizer 抓到）；修好后 10.14ms 反而更慢
  ——单地址共享原子 + flush 不赚钱。
- 固定槽位 cl[12] 寄存器化 claim 收集：8.18ms，更慢（unroll 开销）。
- 手写 barrier / 条件 grid.sync：条件分支里的 grid.sync 会死锁（≥2 block）；
  模板编译期分派 __syncthreads 单 block 版仅省 0.26ms（grid.sync 不是瓶颈）。
- ncu 无权限（ERR_NVGPUctrPERM），靠 nsys + 计时分解定位。

## ws_full 设计（canonical 尾 GPU 化）

三 kernel 接在 sweep 后（同流）：
1. `ws_counts_adj_kernel`（256×256）：per-label 面积（shared 归约+合并）+
   有向邻接 adj[a,b]（a≠0 侧、右+下，散原子，语义与 numba 逐条对齐）。
2. `ws_remap_kernel`（1×512）：并行扫 a，小区域(<32)并入共享边界最长的
   非小邻域，first-max tie 同 numba；非小区域 remap=恒等（**numba 默认
   np.arange 恒等，第一版写成全 0 把标签清光了**）。
3. `ws_apply_boxes_kernel`（256×256）：单遍应用 remap + per-block bbox/area
   shared 归约 → 全局 atomicMin/Max/Add 合并；缺席标签 area=0 由主机端
   补 numba 哨兵（x0=1<<30, x1=-1）。
   **坑：max 累加器 init 必须是最小值 0，"缺席哨兵"只用于缺席（global 与
   shared 两层各踩一次 0xFFFFFFFF 毒化）。**

scratch 约束：adj 为 nl1² int32（nmarkers≤3000 → ≤36MB）；boxes shared
5×nl1×4B ≤ 60KB，3090 (99KB) 内。nmarkers 上限 3000（payload 实测 ~50）。

## 3090 适配（R3 版）

- sweep kernel 无动态 shared、64 寄存器、单 block 常规 launch（不再依赖
  cooperative launch；WS_NBLK>1 才走 cooperative 多 block 备份路径）。
- merge/box kernel shared ≤ 60KB ✓；TORCH_CUDA_ARCH_LIST=8.6 重建即可。

## 已知余量

- host 侧 ~1.6ms（pinned 拷贝 0.14 + H2D 0.36 + D2H 0.31 + pybind/torch
  dispatch ~0.3 + nmarkers GPU max ~0.1）：合并为单 C++ 调用 + 双流 H2D
  可再省 ~0.3ms；K=8 已到 6.38ms。
- 每 wave ~3.5us（hop-1 链：pop→meta→CAS→push）：进一步需 profiler 级
  优化（ncu 权限未开）。
