# team_trt — fwd 轨道：SeedNet 前向替换为 TensorRT（fp16）

k100（RTX PRO 6000 Blackwell, sm_120, 95 GiB）实测；TRT 11.2.1.2 /
onnx 1.22.0 / torch 2.10.0+cu130。ckpt = `/home/k100/gisec_runs/e26/
e26_offw0/runs/ema_ep15.pth`（`{"model": state_dict}`，strict load 通过）。

## 结果一览（全部真实运行）

| 引擎 | harness fwd (3-rep 中位, 40 图) | 下游 AP | delta vs canonical |
|---|---|---|---|
| **seednet_fp16.engine（交付）** | **4.24–4.32 ms/图** | **0.93856** | **+0.00009** |
| 同上, TEAM_TRT_COPY_OUT=0（返回 pinned 视图） | 3.59 ms/图 | 0.93856 | +0.00009 |
| seednet_fp32st.engine（对照） | 7.04 ms/图 | 0.93848 | +0.00001 |
| torch eager fp32（基线） | 23.68 ms/图 | 0.93847 | — |

canonical AP = 0.93847；质量门 > -0.005：**fp16 通过**（甚至略正）。
速度目标 ≤6 ms/图：通过（默认安全模式 4.3，视图模式 3.6）。

引擎纯执行（无 H2D/D2H）：fp16 **1.669 ms**，fp32st/tf32 4.18 ms。
CUDA graph 整段 replay（H2D 7 MB + engine + D2H 4.75 MB）：2.47 ms。
附测吞吐（含整批 H2D/D2H）：bs=4 **2.61 ms/图**（382.6 img/s），
bs=8 **2.53 ms/图**（394.7 img/s）；批引擎与 bs1 引擎 sem 最大漂移
~2.5（fp16 tactic 变化所致，量级与 fp16-vs-fp32 差同阶）。

## 1) ONNX 导出

- wrapper 把 canonical 预处理折进图（`gisec.inference._forward` 语义，
  tensor/tensor IEEE 除法，sub→div→clamp）：
  输入 `img` u8 (1,1024,1024,3)【HWC 原样上船，图内转 CHW】、
  `depth` f32 (1,1024,1024)；rgb/255、(d-0.245)/0.441 clamp[-1,2]、
  concat → (1,4,1024,1024)。
- 输出折进图：`sem_logit` (1,1,1024,1024) f32、`hm` = sigmoid(seed[:,0:1])
  (1,1,256,256) f32、`off` = seed[:,1:3] (1,2,256,256) f32。
- `torch.onnx.export(..., opset_version=17, dynamo=False,
  do_constant_folding=True)`，固定 shape；两图各 **146 节点**，
  fp32 64.3 MiB / fp16 32.2 MiB。ops：Conv×36（**BN 在导出期已折叠进
  Conv 权重，图中无 BatchNormalization 节点**）、Relu×31、Add×8、
  Concat×10、Resize×5、AveragePool×1、MaxPool×1、Cast×4、Clip×1、
  Div×2、Sub×1、Transpose×1、Sigmoid×1、Slice×7。
- 数值：Wrapped(fp32) vs `inference._forward` 逐位一致（max abs
  diff = 0.0）；torch fp16-body vs fp32：sem max 0.82 / hm 9.0e-3 /
  off 2.0e-2（权重 f16 舍入的固有差距）。

## 2) TRT 11.2 的三个坑（重要）

1. **`BuilderFlag.FP16/BF16/INT8` 与逐层 precision API 已被移除**。
   fp16 的唯一路径：ONNX 图里显式承载 half（权重 `.half()`、预处理
   （f32）后 cast、头输出 cast 回 f32），并以 `STRONGLY_TYPED`
   network 解析构建。io 保持 f32，主机侧无感。
2. 执行 API 更名：`enqueue_v3` → **`execute_async_v3(stream)**`。
3. **weakly-typed + `clear_flag(TF32)`（严格 fp32）构建出坏引擎**：
   sem L2 rel 0.30、hm diff ~0.97（输出完全错误），带/不带 timing
   cache 均复现（sm_120 + TRT 11.2.1.2）。weak+TF32=on 与
   strongly-typed 均正确。附带发现：torch cudnn 卷积默认允许 TF32，
   canonical eager 本身就是 TF32-conv。因此**对照引擎取
   strongly-typed fp32 图（fp32st）**：对 torch 默认参考 L2 2.8e-3、
   sem max ~0.9-1.2，AP delta +0.00001。
   （坏的 weak-strict 构建产物已删除；3090/sm_86 上未必有此 bug。）

fp16 引擎 vs torch fp32（canonical 语义）：sem max abs 2.6-3.4
（L2 rel 8.8e-3）、hm max ~2e-2、off max ~8e-2 —— 全部落进下游 AP
+0.00009 的可接受域。

## 3) solution.py 数据通路

```
np.copyto 進 pinned 平面缓冲 (7MB)          ~1.0 ms  (host memcpy)
CUDA graph replay: H2D → TRT → D2H          ~2.5 ms  (GPU, 含 PCIe)
(默认) 输出 host copy                        ~0.66 ms
合计                                          4.2-4.3 ms
```

- **惰性初始化**：首次 fwd 反序列化引擎（~0.2 s）+ 3 次 warmup +
  CUDA graph 捕获；构建/初始化均在计时外（harness 3-rep 中位）。
- 输入/输出各驻留一对 pinned/device **平面字节缓冲**（4K 对齐偏移，
  `set_tensor_address` 指向偏移处）→ 每方向仅 1 次 memcpy。
- **整段 GPU 序列捕获为 CUDA graph**（所有地址跨调用固定，仅内容变；
  TRT 需先 warmup 再捕获；捕获时必须把 capture stream 传给
  `execute_async_v3`，否则 TRT 计算不入图——已踩过，输出会是垃圾）。
  submit 从 ~0.9 ms 降到 ~0.02 ms。
- 默认返回安全拷贝；`TEAM_TRT_COPY_OUT=0` 返回 pinned 视图（下次
  fwd 前有效，3.59 ms）。`TEAM_TRT_ENGINE` 可切换引擎，
  `TEAM_TRT_GRAPH=0` 退回逐 op 派发路径。
- 返回值：sem_logit (1024,1024) f32、hm (256,256) f32、off (2,256,256)
  f32，dtype/shape 与 canonical 一致。

## 4) 显存（实测）

- 引擎运行（仅加载 solution 的进程）：fp16 **976 MiB**
  （nvidia-smi 进程总量，含 CUDA context ~0.5 GiB + 引擎权重 33 MB +
  激活），torch allocator 峰值仅 **12.0 MiB**（平面 IO 缓冲）；
  fp32st 1132 MiB。harness 报 peak VRAM 0.01 GiB（torch 口径）。
- 引擎构建：builder workspace 上限 6 GiB
  （`config.set_memory_pool_limit(MemoryPoolType.WORKSPACE, 6<<30)`）；
  构建期整机 GPU 峰值实测 8.16 GiB（含 build 后 verify 阶段的 torch
  参考模型 3.4 GiB）。24 GiB 预算内富余。

## 5) 3090 适配（引擎必须重建）

TensorRT 引擎与 GPU 架构绑定：本目录 `.engine` 为 sm_120 产物，
**在 3090（sm_86）上必须重建**；`.onnx` 可移植（也可在 3090 机上
重跑导出）。步骤：

```bash
# 3090 机器, 同 TRT 11.x (strongly-typed 需要; 已验证 11.2.1.2)
python export_onnx.py                 # 或直接拷 seednet_fp16.onnx
python build_engine.py fp16           # 产出 seednet_fp16.engine (~20 s)
python harness.py fwd ../team_trt/solution.py
```

builder 参数（build_engine.py 已固化）：`STRONGLY_TYPED` network +
fp16-cast ONNX；workspace 6 GiB；optimization level 默认 3；静态
shape（1×4×1024×1024，无 profile/DLA）；共享 timing cache
（cache.trt，可删）。预期：fp16 引擎本体在 3090 上约 2-3.5 ms
（fp16 tensor 吞吐差 ~3-4×，PCIe gen3 x16 拷贝成本与本机相近），
全链 ~5-7 ms/图——需实测；AP 门在重建后复测一次（fp16 tactic 选择
依卡而异，本卡上 +0.00009 的余量不大可能有质变）。

## 6) 文件清单（均在 team_trt/）

- `solution.py` 交付入口（`fwd`）
- `export_onnx.py` 导出 fp32/fp16 ONNX + 逐位校验
- `build_engine.py` 构建/校验/测延迟（fp16 / fp32st / tf32）
- `bench_solution.py`、`profile_fwd.py`、`vram_probe.py`、`bench_batch.py`（bs=4/8）
- `seednet_fp32.onnx`、`seednet_fp16.onnx`、`seednet_fp16.engine`、
  `seednet_fp32st.engine`、`seednet_tf32.engine`、`seednet_fp16_b{4,8}.onnx/.engine`

未采用的备选：sem 输出降 f16（D2H 省一半，~0.5 ms，阈值 2.944 附近
有翻转风险）；depth 上传 f16（改变预处理语义，否决）。
