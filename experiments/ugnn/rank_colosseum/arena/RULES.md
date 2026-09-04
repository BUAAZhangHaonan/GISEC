# Rank Colosseum — 规则与接口

目标：`gisec.postproc_fast` 的三个排序函数是推理链最大瓶颈（实测 sem_logit_rank
210ms + mix_elevation_rank 162ms + 冷态 depth rank 189ms / 图）。替换排序实现，
其余一律不动。

## 队伍接口（你的 solution.py 必须实现）

```python
def rank_sem_logit(sem_logit):  # (1024,1024) f32 -> (rank int32 (H,W), nrank int)
def rank_mix(rank_d, rank_s):   # 两个 (H,W) int32 rank 图 -> (rank int32, nrank int)
def rank_depth_cold(depth):     # (1024,1024) f32 -> (rank int32, nrank int)
```

## 语义（必须逐位复刻）

- `_rank`：tie 共享 rank（组号），nrank = 不同值的个数。参考实现在
  `gisec/postproc_fast.py` 的 `_rank` / `sem_logit_rank` / `mix_elevation_rank`
  / `compute_elevation_rank`。**先读源码。**
- **关键数学事实（你们要利用的）**：rank 数组对"tie 内部先后顺序"不敏感
  （scatter 写入的是同一个组号），所以任何正确的排序算法（稳定或非稳定）
  都产生逐位相同的 rank 数组。这是本竞技场允许换算法的理论依据。
- **硬性红线：sobel/hypot 必须直接调用参考模块的 `_sobel_xy` / `_hypot_f32`**
  （`from gisec.postproc_fast import _sobel_xy, _hypot_f32`），保证海拔值本身
  逐位不变。只允许替换"值 -> rank"的排序段。混用自算梯度 = 直接判负
  （浮点低位翻转会改变近 tie 的排序）。
- f32 排序键变换注意 `-0.0`：IEEE 里 -0.0 == +0.0 必须共享 rank（numpy
  语义），位变换前要先归一 -0.0 -> +0.0。
- mix 输入是有界 int64（rank_d + 2*rank_s，值域 < 3e6），适合计数/基数排序。
- 输出 rank dtype 必须是 int32，nrank 是 int。

## 验证与计时

```bash
cd <arena 目录>
/home/k100/miniconda3/envs/gisec/bin/python harness.py check ../team_X/solution.py
/home/k100/miniconda3/envs/gisec/bin/python harness.py bench ../team_X/solution.py
/home/k100/miniconda3/envs/gisec/bin/python harness.py refbench
```

- check：40 张真实 val payload + 覆盖 tie/-0.0/退化形状的 fuzz，全对才算过。
- bench：20 张图，每函数 median-of-3。参考基线见 refbench。
- 环境纪律：GPU 空闲可用；任何 >2min 或吃内存的运行必须放
  `systemd-run --user --unit=<名> -p MemoryMax=24G --wait -- 绝对python ...`
  下跑。禁止改动 `src/gisec`（只写自己的 team 目录）。
- 产出：`solution.py`（自包含，可 import numba/torch，但 torch 只许惰性 import）
  + `NOTES.md`（思路、check/bench 输出贴文、集成注意事项）。

## 裁判标准

1. check 全过（否则出局）；2. sem+mix 毫秒数；3. 退化输入稳健性；
4. 集成成本（能否原地替换 `postproc_fast` 内部；CPU-only 依赖是否干净；
   fork worker 里能否用）。
