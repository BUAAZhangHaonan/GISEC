# team_fuse — 融合与流水线工程（C 档极限竞技场）

## R3 增量（solution_trt.py：TRT fp16 前向入图，2026-09-05）

用 team_trt 的 `seednet_fp16.engine`（strongly-typed fp16，预处理+hm-sigmoid
已折进引擎，io f32）替换图内 compiled 前向，整段
**pinned H2D(7MB,1次) → TRT → 编译后处理链(NMS/二值化/sobel/rank/mix) →
pinned D2H(sem 1MB + rank 4MB + marker/cnt/nrank 小包)** 捕获为**一个**
CUDA graph replay。接口与 R2 集成完全一致（`stage(img,depth)->payload dict`），
`solution.py` 未动。

| 指标 | R2（default 编译前向） | **R3（TRT 入图）** |
|---|---|---|
| GPU 段 wall/图（本机无争用实测） | 8.53 ms（R2 集成环境 11.0） | **4.91 ms** |
| 图回放 GPU 时间（event） | 5.6 ms | **3.01 ms**（H2D 0.4 + TRT 1.67 + 后处理 ~0.7 + D2H 0.25） |
| harness fwd 轨道 | 6.65 ms（autotune 配置） | **4.38 ms（5.4x vs 23.7 基线）** |
| 整链 AP delta（40 图，canonical CPU 尾） | −0.00000 | **+0.00009**（n_pred −1/40，marker 数差 4/40，fp16 边界峰翻转） |
| harness fwd AP delta | −0.00099 | **+0.00009** |
| 一次性初始化（计时外） | 6–8 s | **4.0 s**（引擎反序列化 0.2 + 后处理链编译 ~3.4 + 双图捕获 <0.1） |
| 显存 | 0.89 GiB | **torch 池 0.06 GiB**；nvidia-smi 进程 734 MiB（引擎+context） |

单车道整链（stage 与 1 线程 CPU 尾重叠，本机）：67.6 ms/图（14.8 img/s），
地板仍是 CPU 分水岭尾；GPU 段本身 4.91 ms。R2 集成环境（其 CPU 尾 ~23 ms）
预计串行整链 ≈ 28 ms/图量级、吞吐相应抬升。

**TRT 入图要点（实测复验）**：
1. 捕获前必须先在旁路 stream 上做 3 次 warmup enqueue（TRT 惰性模块初始化）；
2. `ctx.execute_async_v3()` 必须传**捕获 stream** 的 ptr——`torch.cuda.graph()`
   上下文里就是 `torch.cuda.current_stream().cuda_stream`，传错则计算不入图
   （team_trt 踩过的坑，复现确认）；
3. 所有地址跨 replay 固定：扁平 pinned/设备字节缓冲（4K 对齐偏移 +
   `set_tensor_address` 指向偏移），TRT 输出即后处理链的编译输入（同一块
   device 内存的 torch 视图），零额外拷贝；
4. 同一个 TRT context 捕获**两张图**（stage 大图 + fwd 小图）可行，互不干扰；
5. payload 的 D2H（sem/rank/小包）也放进图内（memcpy 节点），每图仅剩
   np.copyto 输入 + 一次 replay + 一次 stream 同步；
6. 后处理链用 mode="default" 编译即可（无卷积，autotune 无收益）；
7. fp16 语义注意：sem_logit 粒度变粗 → nrank 从 ~844k 降到 ~578k（ties 变多，
   rank 链仍精确），hm 漂移 ~2e-2 使 4/40 图 marker 数 ±1——AP 端 +0.00009
   （裁判口径一致）。

R3 自测命令：
`systemd-run --user --unit=fuse-r3 -p MemoryMax=24G --wait --
/home/k100/miniconda3/envs/gisec/bin/python bench_trt.py all`；
harness：`cd arena && python harness.py fwd ../team_fuse/solution_trt.py`
（FUSE_TRT_ENGINE 可换引擎；3090 上需用 build_engine.py 重建引擎后复测）。

---

赛道：把 `gisec.gpu_pipeline.gpu_stage`（本机实测 **17.2 ms/图** wall，GPU busy
11.4 ms，374 次 kernel launch + 17 次 memcpy）压到极限：inductor 编译整段
marker 无关 GPU 段 + 手工 CUDA Graph 捕获 + 图内 NMS + pinned 单同步下载 +
批处理吞吐模式。所有数字为 k100（RTX PRO 6000 Blackwell, 188 SM）真实实测，
med-of-N（N≥30 stage / ≥120 全集），预热 ≥5；`torch.cuda.set_per_process_memory_
fraction(24.0/97.9)` 全程强制。注意：本机 GPU 与其他队共享，最后几轮吞吐
复测时撞上他人 100% 占用的调试作业（`scratch/dbg*.py`），受污染的读数已弃用，
下表只引用无争用窗口的数字。

## 1. 组件耗时表（每图）

| 段 | 基线 gpu_stage | team_fuse (default) | team_fuse (max-autotune) |
|---|---|---|---|
| GPU 段 wall（含 copy-in/H2D/D2H/同步） | 17.2 ms | 8.53 ms (2.01x) | **7.66 ms (2.25x)** |
| 图回放 GPU 时间（event 计时） | 11.4 ms busy（374 kernel） | 5.6–5.7 ms（209–227 kernel） | **4.89 ms** |
| kernel launch 开销 | ~6 ms 间隙 + 374 次 | ~14–20 µs/回放 | 同左 |
| fwd 接口 wall（harness 口径） | 23.7 ms（arena 基线） | 8.09 ms | **6.65 ms (3.56x)** |
| 一次性编译（排除在计时外） | — | 6–8 s | 123 s（inductor cache 后秒级） |
| 显存峰值 | 0.80 GiB | 0.89–0.98 GiB | 0.89 GiB（fwd 轨道 0.56） |

基线 17.2 ms 的去向（profiler）：BN-eval kernel 3.5 ms + cudnn NCHW↔NHWC 内部
转置 ~1.8 ms + pageable D2H 0.75 ms + 卷积 ~2.9 ms + ~6 ms launch 间隙。
team_fuse 后：BN 全折进卷积/逐点核、逐点链 triton 融合、整个静态段一次
graph 回放（14 µs 提交）、pinned 单次 event 同步下载 sem(1MB)+rank(4MB)+
markers/cnt/nrank（小包合并）。

图内各段（max-autotune 图，bs=1）：前向 ~3.9 ms（仍含 ~1.3 ms convertTensor
转置，见坑 8）、NMS(cumsum+scatter) ~0.35 ms、双 sobel+rank（f64 精确语义）
~0.5 ms、mix rank ~0.2 ms。

## 2. 端到端与吞吐

单图延迟（GPU 段）：**7.66 ms**；整链单图（GPU 段 + 1 线程 CPU 尾重叠，即
gpu_fast 的 threaded 模式）：**70.6 ms/图（14.2 img/s）**——地板是 CPU
watershed 尾（本机实测 80.9 ms/图，与 eager gpu_stage payload 的 84.8 ms/图
同量级；arena 基线 46.4 ms 只计 _ws_bucket+_merge，不含 boxes/RLE/COCO dict）。

吞吐模式（批前向 + 图内逐行 NMS/rank + 4 线程 numba 分水岭，pinned 4 槽轮转）：

| 配置 | 端到端 | GPU-only 流 |
|---|---|---|
| bs=1, 1 worker | 14.2 img/s（70.6 ms/图） | 6.95 ms/图 |
| bs=4, 4 workers | 14.8 img/s（67.3 ms/图） | ~8 ms/图 |
| bs=8, 4 workers | **15.7 img/s（63.9 ms/图）** | **8.0 ms/图（≈125 img/s GPU 余量）** |
| bs=8, 8 workers | 13.7 img/s（更差：内存带宽争用，单作业 81→119 ms） | — |

吞吐天花板 = CPU 尾 / 4 线程（≈16 img/s）；GPU 侧批处理只摊薄了卷积
（2.4→1.4 ms/图），convertTensor 转置与逐点算子随 bs 线性放大，所以 GPU-only
批流 ~8 ms/图 ≈ 单图。GPU 分水岭（ws 队）落地后此流水线即为其现成前级。

## 3. 质量门（40 payload，canonical CPU 尾）

| 配置 | 整链 segm AP | delta | n_pred | marker 数差 |
|---|---|---|---|---|
| FUSE_MODE=default | 0.93847 | **−0.00000** | 1941 (=1941) | 0/40 |
| FUSE_MODE=max-autotune-no-cudagraphs（默认） | 0.93748 | **−0.00099** | 1941 | 0/40 |
| harness fwd 轨道（default） | 0.93847 | −0.00000 | — | — |
| harness fwd 轨道（max-autotune） | 0.93748 | −0.00099 | — | — |

官方结果：`FWD-RESULT team_fuse: fwd 6.65 ms/img (torch eager 23.7) | AP
0.93748 vs canonical 0.93847 (delta -0.00099) | peak VRAM 0.56 GiB`。

漂移来源量化（fwd 输出 vs canonical 存档）：TF32 本身就在 canonical 链里
（TF32 vs FP32 差 2.4 logit）；inductor 卷积选择 vs eager cudnn ≈ 0.09 logit /
0.0014 hm；BN 手工折叠再 ×7（0.46 logit）——折叠已被关闭（见坑 9）。mix rank
与 eager gpu_stage 的像素级相等率 ~0.0005（任何 logit 噪声都会整体置换稠密
rank，属预期），但序结构几乎不变，AP delta 如上。sobel/hypot 保留 f64 逐步
舍入的精确语义（与 numba 参考位级同构），rank 用 int32 精确混合。

## 4. CUDA Graphs / torch.compile 兼容性坑清单

1. **非zero/动态形状不能进图**：NMS 改写为 raster cumsum + 定容 512 scatter
   （与 canonical 解码 10/10 位级一致）；>512 溢出单图走 eager 精确回退
   （40 payload 上从未触发），批模式直接报错（raster-前-512 与 top-512-by-
   value 语义不同，宁可失败不可错）。
2. **inductor 一维 scatter + 计算索引（arange+where 树）codegen 崩溃**
   （"list indices must be integers"）→ 改 per-row `scatter(dim=1)`。
3. **max-autotune 下批 NMS 被融成单个 split-persistent kernel + workspace，
   bs=8 时 illegal memory access**（torch 2.10.0+cu130 / triton 3.6.0）→
   批函数固定用 mode="default" 编译（`FUSE_BATCH_MODE`），单图段用
   max-autotune。
4. **图输出活在图私有内存池**，下次回放被覆写：payload 消费必须即时/拷贝/
   轮转（stage 返回 sem/rank 拷贝；fwd 8 槽 pinned 轮转视图；批 4 槽）。
   曾因共享 pinned 视图 + nrank 差一导致 `head[nrank]` 越界写、堆损坏
   （`free(): invalid next size`），排查良久。
5. `.item()`/int() 同步全部移出图：nrank/cnt 以张量随小包 pinned 下载，
   每图仅一次 stream.synchronize()。
6. `torch.sort(stable=True)`/cumsum/scatter/replicate-pad/max_pool 均可捕获；
   cub radix 临时内存在图池内复用，无需处理。
7. **cudnn.benchmark 捕获期可用**（algo 锁定），但 autotune workspace 把
   显存顶到 18.7 GiB——3090 上有 OOM 风险，只省 0.13 ms，弃用。
8. **convertTensor 转置杀不完**：channels_last 输入+权重、layout_optimization
   开关、BN 折叠、max-autotune 都试过，cudnn 引擎选择仍留下 ~65-67 个
   转置/回放（1.3–1.5 ms/图，bs 线性放大）。max-autotune 的 triton 卷积模板
   能吃掉一部分（5.6→4.89 ms），这是 fwd 剩余最大的单点。
9. **inductor freezing 与手工 BN 折叠位级等价**且都正确；但折叠权重 f32
   舍入把漂移放大 7 倍，而 inductor 本来就把 BN 融进逐点 epilogue（折叠
   不省时间）→ 默认关闭。开启后 offw0 未训练 offset 头的解码漂移会让 marker
   落点碰撞，AP −0.0178（gate 外），这是"折叠默认关"的直接原因。
10. 图回放确定性：10 连回放位级一致；两个图（fwd 图 / stage 图）输出互不
    干扰（各自私有池）。

## 5. 自查过的 bug（如实记录）

- `_dense_rank` 丢 `+1`：nrank 少 1 → `_ws_bucket` `head[nrank]` 越界写 →
  间歇性堆损坏（有时侥幸不崩）。修复后 invariant `rank.max() < nrank` 常驻
  断言于 sanity。
- NMS `cnt = idx[-1]`（cumsum−1 的末元素）少 `+1`：每图恰丢最后一个 raster
  marker（40/40 图 n_pred −1，AP −0.0178 的真凶，非漂移）。probe 对比
  canonical 10/10 一致掩盖了它（probe 里的公式写对了，进 solution.py 时丢）。
- 批 scatter 一维索引 inductor 崩溃（坑 2）、bs=8 autotune 融合核越界（坑 3）。

## 6. 3090 适配注记

- 全部数字来自 Blackwell（188 SM、~2x 3090 的算力/带宽）。3090 上图回放
  预计 ~2x：GPU 段 wall 约 10–12 ms/图（对 arena 标称 21 ms 基线仍 ~2x）；
  转置/sort/逐点全随 SM 数缩放，无架构特殊依赖（无 Blackwell 专属指令路径，
  triton 卷积模板在 Ampere 上会重新 autotune）。
- 显存：峰值 3.97 GiB（bs=8 含 4 槽 pinned），fwd 轨道 0.56 GiB，3090 24GB
  余量充足；per-process fraction 限制已内置。坑 7 的 cudnn.benchmark 在 3090
  上更容易顶爆，保持关闭。
- 编译产物（inductor cache）与 autotune 选择是机器绑定的：3090 首跑需重编译
  （default ~10 s，max-autotune ~2–3 min），规则允许排除计时但需上报。
- CPU 尾与 GPU 型号无关；吞吐模式的地板仍是 4 线程 numba 分水岭（~16 img/s）。

## 7. 交付物

- `solution.py`：`fwd()`（fwd 轨道接口）、`FusedStage.stage()`（GPU 段，
  GpuPayload 同构 dict）、`FusedStage.batch_stage()`（吞吐）、`cpu_stage()`
  （canonical 尾）。环境开关：`FUSE_MODE`（default/max-autotune-no-cudagraphs/
  eager）、`FUSE_BATCH_MODE`、`FUSE_FOLD`、`FUSE_CL`、`FUSE_SOBEL`。
- `pipeline_bench.py`：`timing` / `quality` / `throughput` 三模式基准。
- 复现：`systemd-run --user --unit=fuse-X -p MemoryMax=24G --wait --
  /home/k100/miniconda3/envs/gisec/bin/python pipeline_bench.py all`
  （harness：`cd arena && python harness.py fwd ../team_fuse/solution.py`）。

一句话：整段静态 GPU 段一个 CUDA Graph 回放 + NMS 进图 + pinned 单同步 IO，
17.2 → 7.66 ms/图（2.25x，fwd 轨道 23.7 → 6.65 ms，3.56x），AP delta
−0.00099（default 配置 −0.00000），吞吐 15.7 img/s（CPU 分水岭地板）。
