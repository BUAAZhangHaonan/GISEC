# Output Hygiene And Training Observability Design

**Context**

`output/` 已经累积了大量早期 `probe / smoke / v3 / query / bridgegap / purity06 / tailattach` 产物，当前真正还有参考价值的只剩正式 baseline、当前主线 merge pilot 和汇总结论文档。与此同时，训练脚本虽然大多已经只保留 `model_best.pth` 与 `model_final.pth`，但没有统一的“只保留这两个权重”的硬约束，也缺少可直接监控训练效果变化的过程可视化。

**Goal**

把 `output/` 清理成只保留当前主线必需结果的状态，并把主训练链改成：

- 默认只保留 `model_best.pth` 和 `model_final.pth`
- 训练过程中自动写出标量历史图
- 训练过程中自动写出可直接浏览的可视化预览

**Approved Scope**

本次按保守清理方案执行：

- 保留 `output/analysis` 中的正式分析文档
- 删除 `output/analysis/eval_profile_overlays_tmp*`
- 保留 `output/experiments/baselines` 中仍有主线参考价值的正式阶段目录
- 删除 `output/experiments` 下明显失效的 `smoke / probe / v3 / query / bridgegap / purity06 / tailattach` 历史实验目录
- 为 `baseline/unet/train.py` 增加训练过程标量图与 overlay 进度快照
- 为 `baseline/reference_graph/train.py` 增加训练过程标量图与 merge 预览快照
- 为主训练链加入 checkpoint 清理步骤，硬保证只留 `model_best.pth` 与 `model_final.pth`

**Design**

1. Output hygiene

新增一个可复用清理脚本，按 allowlist 保留仍有参考价值的实验目录，其余目录统一删除。脚本同时清理 `output/analysis` 下的临时 overlay 目录。这样后续每次清理都有可复用入口，不需要手动 `rm -rf`。

2. Checkpoint retention

新增一个共享训练产物辅助模块，提供“删除除 `model_best.pth` / `model_final.pth` 之外所有 `.pth` 权重”的函数。`unet` 和 `reference_graph` 训练在保存完 best/final 后统一调用，确保以后不会因为局部改动引入额外权重残留。

3. Training observability

新增共享训练可视化辅助模块，提供：

- `history.jsonl` 逐次记录训练/验证标量
- `training_curves.png` 折线图，方便直接看 loss / AP / F1 / threshold 演化
- 训练过程 preview contact sheet

`unet` 训练直接复用 eval 已经产出的 overlay，并在每次 eval 后快照成按 epoch 命名的进度预览。

`reference_graph` 训练新增 merge preview 渲染：对验证集前若干张图，加载 query image，渲染 `fragments -> merged` 对比图，并在每次 val 后写出 contact sheet，便于直接看 merge 是否开始过并、欠并或改善。

**Non-Goals**

- 不改动现有 benchmark 指标定义
- 不清理当前仍在使用的正式主线实验目录
- 不把 tail-attachment 这类已证伪策略合入主线默认逻辑

