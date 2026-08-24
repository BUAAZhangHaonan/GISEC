# E17: 边界带加权 BCE + EMA（零新参数）

## 状态

- [x] build_band_records.py（band 预计算，见下方偏差说明）
- [x] train_band_ema.py（fork exp10 train_capacity.py，仅改 band BCE + EMA + 锁钩子）
- [x] 冒烟（50 step + 2 val batch，32G cap 独立单元）
- [x] 正式启动：systemd 单元 `ugnn-e17-train`，20ep from scratch，out-dir runs/
- 锁：/tmp/gisec_gpu_priority 由训练脚本 --lock-file 创建/finally 删除；与 baselines16m 链的 mrcnn16 并行双跑，mrcnn16 结束后链暂停等 E17

## 改动（相对 E10 canonical）

1. **band 加权 BCE**：weight = 1 + 3·band，band = 逐实例 dilate(m,3×3) & ~erode(m,3×3) 的并集（缝隙两侧 + 外轮廓）。仅 BCE 项加权，Dice/focal/offset 不变。
2. **EMA**：decay 0.999 每步更新（float 张量平均、int buffer 拷贝）；val 用 EMA 权重 swap in/out 评一次、raw 评一次，best.pth 存 EMA state，best 按 EMA mIoU。

参数量断言 = 16.851M（无新头）。评测期全部不动（HM_THR .3 / SEM_THR .6 / 峰值打分 / mix λ=2）。

## 实现偏差（相对任务书）

任务书要求 worker 内逐实例算 band，但 gt_records 的 items.pkl **没有**逐实例 1024 mask（只有 union sem + stride-4 inst4 + (fy,fx,n) 统计）；worker 内解原始标注正是 exp09 build_gt_records.py 要消除的 COW 内存坑。故改为一次性预计算（16 进程，模式同 exp16 build_flow_records.py）：`gt_records/{split}_band.dat` packbits，行序与 items.pkl 对齐（构建时断言校验），band 定义逐字不变，训练 loader 零额外开销（优于 100-150ms/样本预算）。

## 预注册（同步写入 RESULT.md 头部）

- PASS：训完后 500 图配对 CI（E17 best ckpt vs E13 行 0.81503，同图配对）ΔAP>0 且 CI 不含 0，且全量 fast FINAL > 0.82137。
- 护栏：种子 median <8px（防 E17 伤种子）。

## 冒烟数字（2026-08-24 03:07，与 mrcnn16 并行争用下）

- 参数量 16.851M（enc 11.180 + dec 5.610 + seg 7993 + seed 52803），零新增 ✓
- 50 step 末 loss：bce 0.4756 / dice 0.3082 / focal 1.5280 / off 0.5059，全有限 ✓
- band_frac 0.0399，带内平均权重 w_in_band = 4.00（=1+3·1）✓
- smoke val（2 batch）：raw 0.8155 / EMA 0.2899（decay 0.999 下 50 step EMA 仅吸收 ~5% 权重，滞后是预期行为，swap 双向正常）✓
- 速度：E17 ~0.60 s/step vs 同条件 fork 前（exp10 原脚本 50 step 实测）~0.50 s/step，劣化 ~20% < 30% 预算 ✓
- band 预计算：val 441s + train 3176s（16 进程），spot check band ⊆ dilate(sem,3×3) 通过
- 冒烟产物（runs_smoke / exp10 runs_e17_cmp）已清理

## 运行

- 单元：`ugnn-e17-train`，MemoryMax=160G，CPUQuota=3200%，workers 16
- 预计 ~8-9h（与 mrcnn16 并行争用下），2026-08-24 启动

## full-profile bootstrap（2026-08-24 15:39 启动，无人值守）

- systemd 单元：`gisec-e17-fullboot`（MemoryMax=160G, CPUQuota=3200%）
- 命令：eval_centernet.py --arch e10 --profile full --ckpt runs/best.pth
  --out runs/eval_report_full.json
- ETA ~1-2h；查看：`journalctl --user -u gisec-e17-fullboot -f`
- fast 全量数字已出（AP 0.83808），full 只补 oracle/seed/bootstrap CI。
