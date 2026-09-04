# team_b — numba 基数/计数排序 rank

## 思路

三个函数只替换"值 → 名次"段；梯度/幅值段逐字复用参考实现
（`from gisec.postproc_fast import _sobel_xy, _hypot_f32`，红线遵守），
因此喂给排序器的浮点位与参考完全一致。

### f32 rank（`rank_sem_logit` / `rank_depth_cold`）

1. **保序 u32 键变换**（IEEE total order）：先把 `-0.0` 位型
   `0x80000000` 归一为 `0`（`-0.0` 与 `+0.0` 必须共享 rank），负数按位取反、
   非负数置最高位。浮点序 == u32 键序，相等浮点 == 相等键。
2. **(key<<32 | index) 打包成 u64**，做 **3 趟 11-bit LSD 基数排序**：
   每趟只重排这个 packed 数组——全程顺序读、每元素仅一次随机 8 字节写
   （两数组分开散写会是两次随机行触碰，打包是关键提速点）。
   tie 共享组号且 rank scatter 顺序无关 ⇒ **不需要稳定排序**，桶内填充
   顺序任意。某趟数字全常数则跳过该趟（零图/常图/斜坡 fuzz 直接退化）。
   每线程 8KB 私有直方图（L1 常驻），total/cursor 簿记用**行优先**
   （列优先的跨步访问实测把一趟拖到 9-14ms，行优先后整趟 ~1.5-2ms）。
3. 排序完成后 keys 本身已顺序 ⇒ 分组 = 顺序 flag 扫描 + 串行 inclusive
   前缀 + 并行 scatter 回原位置。

### mix rank（`rank_mix`）

值域有界非负 int（`rank_d + 2*rank_s`，实测 K≈2.7e6），**O(n+K) 计数
排序直出 rank，完全无比较排序、无置换**：

1. 并行填 `v32` 并求 min/max（int64 算术，防 int32 溢出判断）。
2. **每线程位平面（bit-plane）标记**：`planes[t][v>>6] |= 1<<(v&63)`，
   行私有 ⇒ 零伪共享（共享 int32/byte 数组的随机标记在这个多 socket
   机器上因 cache 行争用反而更慢，实测 2-4ms → 位平面 ~0.5ms）。
3. OR 合并成全局位图，SWAR popcount 两级块扫描得到每 64 值块的
   base 计数；`rank(v) = wbase[v>>6] + popcount(comb[v>>6] & 低位掩码)`
   —— 查表全部落 L2（位图 ~340KB），没有 11MB 稠密 off 数组。
4. 负值 / 值域 ≥ 2^26 的退化输入直接回落参考实现
   （`mix_elevation_rank` 本身，语义逐位一致；harness 用例不会触发）。

### 工程要点

- **单用途 kernel**：最初把多趟 radix 写进一个大 `parallel=True` 内核，
  在 numba 0.67 上被 parfors 错误编译直接段错误；拆成"每 kernel 一个
  阶段、Python 端编排"后稳定。
- **`get_num_threads()` 不能出现在 jit 体内**（dynamic global，会让
  cache=True 失效、每个进程重编译 ~25s）；改为 Python 端取 T 传参，
  所有 kernel（含 parallel）都能落盘缓存，热导入 ~1s。
- **scratch 缓冲复用**：每调用 ~26MB 新分配的首触页错误就要 ~10ms；
  按尺寸缓存缓冲（dict + 整个计算持锁，numba kernel 释放 GIL），输出
  rank 恒为 fresh copy（调用方持有旧结果不会被覆写）。
- **线程数**：默认 `min(16, NUMBA_NUM_THREADS)`（本机 128 核，刻意
  克制避免与 torch/BLAS 抢核）；`TEAMB_RANK_THREADS=N` 覆盖，
  `=1` 走纯串行 kernel。16/24 差别在噪声内。
- 串行 kernel 注意点：hist 既当计数又当 scatter cursor，**每趟必须
  归零**（此 bug 曾让串行模式段错误；并行版每趟新分配故不受影响）。

## 结果（原文）

check（并行默认，最终代码）：

```
$ python harness.py check ../team_b/solution.py
CHECK PASS: bitwise identical on all real + fuzz cases
```

check（TEAMB_RANK_THREADS=1 串行）：

```
CHECK PASS: bitwise identical on all real + fuzz cases
```

bench（并行默认，T=16，median-of-3 / 20 图）：

```
$ python harness.py bench ../team_b/solution.py
BENCH team_b: sem    28.1  mix     2.5  depth_cold    27.9  sem+mix    30.6  ms/img
```

（复跑两次分别 28.5/2.3/27.8 与 27.9/2.3/27.5，噪声 ±1ms。）

bench（串行，TEAMB_RANK_THREADS=1）：

```
BENCH team_b: sem    50.4  mix    27.5  depth_cold    48.6  sem+mix    77.9  ms/img
```

参考基线（本机复测 refbench）：

```
BENCH reference(gisec.postproc_fast): sem   208.8  mix   160.7  depth_cold   184.8  sem+mix   369.5  ms/img
```

| 函数 | 参考 ms | team_b ms（并行16线程） | 提速 | team_b ms（串行） |
|---|---|---|---|---|
| sem_logit_rank | 208.8 | **28.1** | 7.4× | 50.4 |
| mix_elevation_rank | 160.7 | **2.5** | 64× | 27.5 |
| compute_elevation_rank（冷态） | 184.8 | **27.9** | 6.6× | 48.6 |
| sem+mix | 369.5 | **30.6** | 12.1× | 77.9 |

分解（并行，1024²）：sobel ~7ms + hypot ~8ms（参考 kernel，串行，不可
动）+ radix rank ~9ms = sem ~28ms；mix：v32 ~0.5 + 位平面标记 ~0.5 +
合并/wbase ~0.6 + scatter ~0.8 + copy ~0.6 ≈ 2.5ms。sem 的地板已由
红线要求的串行 `_sobel_xy`/`_hypot_f32`（~15ms）决定。

## 集成注意事项

- **原地替换**：`postproc_fast` 内 `_rank`/`sem_logit_rank`/
  `mix_elevation_rank` 的排序段可直接换成 `solution._rank_f32` /
  `solution.rank_mix`；`compute_elevation_rank` 保留
  `depth.astype(np.float32) → _sobel_xy → _hypot_f32` 原样（位一致），
  只换 rank 段。CPU-only，无新依赖（numba/numpy 已在用）。
- **编译缓存**：kernel 全部 `cache=True`，产物在 solution.py 同目录
  `__pycache__`；首次编译 ~25-30s（import 时小数组预热，全部排好在
  计时之外），之后热导入 ~1s。部署时保留该目录可写。
- **线程**：默认 `min(16, NUMBA_NUM_THREADS)`；调 `TEAMB_RANK_THREADS`
  （1=纯串行 kernel，适合 fork worker 或避免与 BLAS 打架）。本环境无
  TBB，numba 落 OMP 层；**OMP + fork**：若进程先跑过并行 kernel 再
  fork（如 `precompute_main` 的 `mp.Pool(8)`），建议 worker 内
  `TEAMB_RANK_THREADS=1`，或全局 `NUMBA_THREADING_LAYER=workqueue`。
  先 fork 后调用则无此顾虑。注意 numba 线程池是进程级全局
  （`set_num_threads` 在 import 时调用一次）。
- **内存**：1024² 稳态驻留 ~45MB（rank scratch ~30MB + mix 位平面
  ~5.5MB + grow-only 位图/wbase ~4MB），无每调用大分配；mix 的 grow-only
  缓冲按历史最大 K 保留。fork worker CoW 继承无冲突。
- **并发**：所有共享 scratch 由 `threading.Lock` 整段保护——多线程
  调用时正确但串行化（单图推理链本就顺序调用）。
- **已知语义边界**（本管线不可达，如实说明）：多个不同位型的 NaN 会
  按位型区分名次，且不像参考的 argsort+`!=` 分组那样把相邻 NaN 逐个
  拆组——hypot 输出对真实 payload 恒有限，不触发。mix 输入若出现负值
  或值域 ≥ 2^26 自动回落参考实现（逐位一致）。
