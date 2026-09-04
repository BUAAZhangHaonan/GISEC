# gpu_extreme — C 档极限推理竞技场（2026-09-05，探索性研究）

**不进默认链，不影响 canonical 口径。** 目标：在 24 GiB 显卡（RTX 3090，
Ampere sm_86）预算内把 GISEC 单图推理压到极限。武库：TensorRT fp16 引擎
（入 CUDA graph）、自定义 CUDA 分水岭（cooperative launch）、torch.compile
整段融合 + CUDA graph 捕获。**质量代价如实上报**（非逐位：fp16 前向 +
量化洪泛），验收门 = 全量 val 配对 scene bootstrap delta > -0.005。

## 阶段战报（k100 RTX PRO 6000 实测，E26b ckpt @0.95；裁判复测）

| 轮次 | 配置 | 单图串行 | 吞吐 | AP delta（40 图） |
|---|---|---:|---:|---:|
| 基线 gpu_fast（B 档） | torch eager + CPU numba 分水岭 | ~81 ms 纯算 | ~14 img/s | 0（逐位） |
| R1 三队组件 | TRT fp16 前向 4.28 / CUDA 分水岭 10.18 / 编译融合段 7.42 | — | — | +0.00009 / -0.00023 / -0.00099 |
| R2 集成 | 融合段(default) + ws(K=16) + CPU 尾 | 34.0 ms | 31.3 img/s | -0.00023 |
| R3 组件 | TRT 入整段 graph：stage 4.91；ws 提速 6.6 + merge/boxes 入 kernel（与 numba 逐位一致）= ws_full 7.04 | — | — | -0.00132（K=10） |
| **R4 终集成** | **TRT-in-graph 段 + ws_full + 仅 RLE 的 CPU 尾** | **22.0 ms** | **51.5 img/s** | -0.0009 ~ -0.0018（跨运行非确定） |

## 全量 val 门（3276 图，2026-09-05，arena/full_gate_extreme.py，记录 full_gate_extreme.json）

- 极限链 wall **37.9 ms/img（26.4 img/s）**，其中 IO 冷读 32.1 ms——计算链
  （stage 5.6 + ws 1.3 + 尾 15.4 ms 计时口径，线程重叠后全部藏在 IO 之后）
  已不是瓶颈；**纯计算地板 ≈ 19-22 ms/img（R4 竞技场口径）**。
- **segm AP 0.87325 vs canonical 0.87617，点差 -0.00292；配对 scene
  bootstrap（2000 draws）delta -0.00331，CI95 [-0.00497, -0.00192]**——
  CI 下界 -0.00497 过预注册门 -0.005（**贴线**，余量 3e-5）。如需质量
  余量：`WS_K=16`（ws 8.3 ms，40 图 delta -0.0002）即可把全量 delta 拉回
  ~-0.001 量级，代价 ~2 ms/图。
- 漂移来源分解：ws K=10 量化洪泛（40 图 -0.0013，小目标 APs 掉得多——量化
  层级粗，小件刀口敏感）+ fp16 前向（+0.0001）+ 跨运行非确定性（并行原子
  tie 破解，40 图上 -0.0009~-0.0018 波动）。n_pred/img 52.95 vs canonical
  51.81（K=10 略多切）。
- **显存**：torch 池 0.21 GiB + TRT 引擎/context ~0.75 GB ≈ **1.0 GB**，
  3090 24 GB 预算内富余（构建期峰值 8.2 GB 也 <24）。

## 目录

- `arena/`：规则（RULES_EXTREME.md）、harness（ws/fwd 双轨道，正确性门 =
  下游 AP 等价）、裁判脚本、payload 生成（make_payloads.py，40 图，
  canonical.json 参考含 peaks——注意 E26b offw0 的 offset 头无监督、legacy
  解码落点可跨 cell，peaks 必须存 NMS 源 cell 值）、R4 集成流水线、全量门。
- `team_trt/`：ONNX 导出（BN 折叠、预处理/sigmoid 入图）+ TRT 构建
  （TRT 11 无 FP16 flag → strongly-typed + 图内 fp16 cast；weakly-typed
  TF32 会出坏引擎）+ CUDA graph 回放。**fp16 ONNX 已入库（34MB，可移植）；
  .engine 绑 sm_120 不入库，3090 上用 build_engine.py 重建**。
- `team_ws/`：CUDA 量化分水岭（K 桶升序 minimax 洪泛，cooperative
  launch；R3：warp 聚合 push、seeds 只 claim、去投机 2-hop）+ GPU
  merge/boxes（与 numba 逐位一致，_selfcheck.py 40/40）。
- `team_fuse/`：torch.compile 整段融合（NMS 进图：raster cumsum + 定容
  512 scatter 与 canonical 逐位一致）+ 手工 CUDA graph；`solution_trt.py`
  = TRT 引擎折入同一张图（3.01 ms 图回放）。
- `validate_on_3090.sh`：GPU 5 一键复验（重建 sm_86 引擎 + 扩展编译 +
  40 图双轨道门 + 编译捕获冒烟）。

## GPU 5（4029 / RTX 3090）实测记录

k100 无 4029 凭据（公钥与多组密码均拒），实机复验由用户执行：

```bash
export GISEC_CKPT=<e26_offw0 ema_ep15.pth 路径>
export GISEC_DATA_ROOT=<20260318_1K_32254 路径>
# gisec 环境激活（pip install -e . + pip install tensorrt onnx）
CUDA_VISIBLE_DEVICES=5 bash experiments/ugnn/gpu_extreme/arena/validate_on_3090.sh
```

脚本五步：payload 重建 → sm_86 引擎重建（含数值对拍）→ ws 扩展 native
编译 → TRT-in-graph 捕获冒烟 → 40 图双轨道门 + R4 集成 bench。
**该序列已在 k100 GPU 0 全程干跑通过**（2026-09-05，fwd 4.34ms/+0.00009、
ws 6.69ms/-0.00132、串行 22.0ms/吞吐 19.1ms——与竞技场记录一致）。

| 指标 | k100（Blackwell）记录 | GPU 5（3090）实测 | 日期 |
|---|---:|---:|---|
| fwd（TRT fp16 含传输） | 4.34 ms | 待回填 | |
| ws CUDA 分水岭 | 6.69 ms | 待回填 | |
| 串行单图延迟 | 22.0 ms | 待回填 | |
| 线程吞吐 | 19.1 ms（52.3 img/s） | 待回填 | |
| AP delta（fwd / ws 门） | +0.00009 / -0.00132 | 待回填 | |

预估（README 上文）：卷积/TRT 段 ~2× 慢、kernel 段 ~1×，单图 ~30-40ms。

## 3090 部署注记

- k100（Blackwell）计时对 3090 偏乐观：卷积/TRT 段约 2×，kernel 密集段
  （分水岭/编译图回放）接近 1×。3090 预估：单图 ~30-40 ms、吞吐 ~25-35
  img/s（IO 不变则被 IO 主导）。以 validate_on_3090.sh 实测为准。
- 依赖：tensorrt + onnx（pip，**实验性依赖，不进 pyproject**）、ninja +
  CUDA toolkit（ws 扩展 native 编译——脚本不钉 arch，各主机按本机 GPU
  编译，3090 上即 sm_86）、TRT 引擎须在 3090 上重建。
- 显存红线已在所有 runner 强制（k100 上 set_per_process_memory_fraction
  24/97.9 模拟；3090 上天然 24 GB）。

## 结论（判"已到最佳"的依据）

剩余可挤空间排序：① IO（全量口径 32ms 冷读 >> 计算地板 19ms——机器人/
相机直出张量即消失，非算法问题）；② RLE 编码 ~15ms CPU 尾（可 GPU 化但
pycocotools 生态在 CPU，收益 <5ms）；③ ws 波数收敛结构（~6.5ms，再压需
换算法族）。三者在"探索性研究"定位下收益/复杂度比已不划算，R4 定为当前
最佳。后续若真上 3090 部署，先跑 validate_on_3090.sh 校准。
