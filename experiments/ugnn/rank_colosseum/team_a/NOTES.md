# team_a — 纯 numpy 路线 NOTES

## 一句话思路

浮点路径把「非负 f32 的 u32 位型」和「元素下标」打包进 f64 的 52 位尾数，用被
x86-simdsort 加速的**值排序** `ndarray.sort()` 取代昂贵的 argsort；mix 路径值域有界，
完全免排序：`bincount -> cumsum(count>0) -> take` 三条 numpy 原语完成 O(n+K) 计数 rank。

## 测量背景（本机，numpy 2.4.3，Hygon C86 7285 / Zen1, AVX2）

组件级画像（1M 元素，sem logit 梯度幅度，median 多轮）：

| 操作 | ms |
|---|---|
| `np.argsort(f32, kind='stable')`（参考 `_rank` 用法） | 186.6 |
| `np.argsort(f32, kind='quicksort')` | 81.9 |
| `np.sort(f32, quicksort)`（SIMD 值排序） | 19.5 |
| `np.sort(f64, quicksort)`（52 位打包键） | 33.6 |
| `np.searchsorted(sv, flat)`（1M 查 1M） | 296.6 |
| `_sobel_xy` + `_hypot_f32`（参考 numba，必复用） | ~15–20 |

结论：这台机器上 argsort（含置换跟踪）远慢于值排序；`unique+searchsorted` 路线的
searchsorted 是灾难（300ms）。因此最优路线 = 值排序 + 打包下标。

## 浮点路径（rank_sem_logit / rank_depth_cold）

海拔是 `_hypot_f32` 输出，恒 >= +0.0，无 NaN/-0.0。关键性质：

1. **非负 f32 的 uint32 位型本身就是保序整数键**；`key | 0x80000000` 后仍然保序，
   且把 -0.0 与 +0.0 折叠成同一键 —— 与 IEEE `==`/numpy 比较语义一致（RULES 红线）。
2. 打包 `pk = (key << 20) | index | 0x3FF0...`：key 占尾数 [20,52) 位、下标占 [0,20)
   位（1024×1024 = 2^20 恰好放下）、指数固定 0x3FF → pk 是 [1,2) 内的正常 f64，
   **按值排序 == 按 (key, index) 字典序排序**。
3. `pk.view(f64).sort()` 原地 introsort（无拷贝分配）。打包键两两不同（下标决胜），
   全序严格 → **任何**比较排序结果唯一，稳定性无关（RULES 已论证 tie 共享组号）。
4. 组边界：相邻 `pk >> 20` 比较（屏蔽下标位）；组号 `cumsum(ne, dtype=int32)`；
   `pk &= 0xFFFFF` 后 `view(intp)` **零拷贝**得到 order，散射 `rank[order] = grp`。

守卫与回退（保证退化输入稳健）：
- `n == 0` 早退；`flat.min() < 0`（理论上不可达，hypot 恒非负）或 `n > 2^20`
  → 通用回退：`argsort(kind='quicksort')` + 同样的 cumsum/scatter（仍比参考的
  stable mergesort 快 ~1.8x，逐位正确）。
- ∞/次正规数：正 float 位型保序覆盖所有正位型，天然正确（selftest 覆盖）。

## mix 路径（rank_mix）

`mixed = rank_d + 2*rank_s` 是有界非负整数（1024² 帧 < 3.15M）：

```
mixed = rd + 2*rs                      # int32 原生，无 int64 中间数组
cnt   = np.bincount(mixed)             # 值直方
tbl   = np.cumsum(cnt > 0, i32) - 1    # tbl[v] = 值 v 的 rank（出现值才有效）
rank  = np.take(tbl, mixed)            # 一次 gather，无任何排序
nrank = 恰好 = cumsum 末元素
```

守卫：min/max 四个归约（~1ms）检查非负与 `dmax + 2*smax + 1 <= INT32_MAX`，越界
（负值 / int64 大值 / 溢出风险）→ 通用 argsort 回退。int64 输入自动走回退或安全转
int32。harness 的 3 组 fuzz（small/full nrank、zeros）+ 5 张真实图全部逐位一致。

## 最终输出（原文）

check（40 张真实 val payload + tie/-0.0/退化形状 fuzz）：

```
$ python harness.py check ../team_a/solution.py
CHECK PASS: bitwise identical on all real + fuzz cases
```

bench（同场次先跑的 refbench）：

```
$ python harness.py refbench
BENCH reference(gisec.postproc_fast): sem   213.5  mix   164.1  depth_cold   189.8  sem+mix   377.7  ms/img

$ python harness.py bench ../team_a/solution.py
BENCH team_a: sem    64.5  mix    34.4  depth_cold    64.7  sem+mix    98.9  ms/img (mean of 20 imgs, median of 3 reps)
```

| 函数 | 参考 ms | team_a ms | 加速 |
|---|---|---|---|
| sem_logit_rank | 213.5 | **64.5** | 3.31x |
| mix_elevation_rank | 164.1 | **34.4** | 4.77x |
| compute_elevation_rank (cold) | 189.8 | **64.7** | 2.93x |
| sem+mix（裁判主指标） | 377.7 | **98.9** | 3.82x |

（三次 bench 复跑波动 < ±2 ms：65.5/64.9/64.5 sem。）

## 集成注意事项

1. **原地替换**：可直接替换 `postproc_fast.py` 里 `_rank`（浮点调用侧）、
   `sem_logit_rank` / `compute_elevation_rank` / `mix_elevation_rank` 函数体；
   sobel/hypot 已按要求原样 import 自参考模块，海拔值逐位不变。
2. **CPU-only / 依赖干净**：纯 numpy（无 numba 新内核、无 torch、无 scipy）。
   fork worker 安全：模块状态只有一个按 n 缓存的只读 `(arange|EXP)` u64 数组
   （惰性填充，>16 项自动清空），copy-on-write 共享，无编译态。
3. **numpy 版本**：在 numpy 2.4.3（NEP-50）验证。位打包用的移位以 0-d u64
   **数组**为操作数，避开 numpy 1.x 值基广播把 u32×u64 标量降成 u32 循环的坑，
   1.x/2.x 理论都安全；但整套性能优势依赖 numpy ≥ 2.0 的 x86-simdsort 值排序
   （AVX2 即可）。无 AVX512 的 Zen1 上也已达标；若无 SIMD-sort 回退到普通
   introsort，值排序仍不慢于 argsort，只是加速比缩小。正确性与 numpy 版本无关。
4. **帧尺寸上限**：打包路径要求 `n <= 2^20`（1024×1024 恰好），更大图自动走
   quicksort-argsort 回退（正确、仍快于 stable 参考约 1.8x），无需改代码。
5. **`-0.0`**：浮点路径 `key | 0x80000000` 折叠 ±0.0；mix 为整数无此问题；
   通用回退用比较排序天然把 ±0.0 视为相等。fuzz（含 -0.0 tiny、全 0、常量、
   ramp、非方形、1×1、denormal、inf）自测全过。
6. 内存峰值：浮点路径临时多 ~3×8MB（pk/ne/order）+ 输出 4MB，比参考（int64
   grp/order）还低；mix 峰值 = bincount 表 K×8B（~22MB @ K=2.7M）。
