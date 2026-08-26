# Cleanup Plan 20260826

来源：依赖测绘 agent 产出（2026-08-26）。供执行 agent 做仓库最小化；本文件随最终 commit 保留或删除由执行者定。

### KEEP（活跃必需）
- exp09_centernet_seeds/：eval_centernet.py、postproc_fast.py、train_centernet.py、centernet_gt.py、build_gt_records.py、build_rgb_cache.py、RESULT/STATUS.md、runs/postproc_cache(13G)、cache_rgb(9.7G)。**gt_records/ 7.2G 删除（build_gt_records.py 可再生）**
- exp10_semantic_capacity/train_capacity.py（SeedNetE10 定义地）→ 迁入 lib/后删目录
- exp08_scale_32254/eval_scale.py（用符号 DATA,HM_THR,SplitStats,gt_center_markers,gt_centers,rss_gb,scene_key,seed_precision,load_split）→ 迁 lib/
- exp03_unet_dense/：eval_pipeline.py(DEPTH_LO/HI,load_depth_array,LiteCOCO,ann_to_mask)、train_unet.py(DEPTH_HI/LO,DenseDataset) → 迁 lib/
- exp04_instance_split/eval_watershed.py（elevation_map,postprocess 被 eval_scale import）→ 迁 lib/
- exp17_band_ema/：build_band_records.py + gt_records/(3.6G canonical 训练数据，含 band) + RESULT.md；其余删（train_band_ema.py 在 git 历史）
- exp20_band8/：train_band8.py、runs/best.pth、runs/train_log.json、RESULT.md；runs/last.pth 与 _cache_fwd 删
- baselines16m/：全部代码+RESULT+STATUS（m2f16fix 训完评完后再删四个 family 的 .pth，保留 metrics.json）
- src/gisec 白名单：__init__.py、config/{__init__,variants}.py、datasets/{__init__,coco_utils}.py、eval/{__init__,coco_eval,coco_export}.py；tests/test_coco_export.py + conftest.py
- docs/ 两份 md、LEDGER.md、datasets/20260318_1K_32254(211G 主数据)

### DELETE
- output/ 整目录 551M；build/、src/gisec.egg-info/、scripts/（M2F CLI）
- src/gisec 可删模块：backbones/、models/、train/、cli/、engine.py、geometry.py、__main__.py、datasets/{baseline_instance_dataset,reference_bank}.py、eval/{boundary_metrics,export,split_merge}.py；tests/ 其余 13 个
- 整目录：exp01、exp02、exp04_fragment_gnn、exp05、exp06(110M)、exp07(110M)、exp11(17M)、exp12(3.3G)、exp13、exp14(6.1G)、exp15、exp16(3.4G)、exp18(3.1G)、exp19、exp21(3.1G)、exp22(3.1G)、postproc_colosseum(2.6G)、heatmap_colosseum
- 部分删：exp03 仅留 eval_pipeline.py+train_unet.py 后删 runs(110M)；exp08 仅留 eval_scale.py；exp10 迁 train_capacity.py 后整删；exp09/{gt_records 7.2G, runs/best.pth 55M}；exp17/{_cache_fwd 3.0G, runs/, train_band_ema.py}；exp20/{_cache_fwd 3.0G, runs/last.pth}
- datasets/20260318_1K_1566(7.2G)、20260318_1K_13440(21G)（旧数据集，实验数字已记录）
- baselines16m 四个 family 的 .pth 权重（m2f16fix 评测完成后）
- exp10/exp19/exp21/exp22 的 gt_records 符号链接随目录删

### ARCHIVE
- 删除前把各历史 exp 的 RESULT.md/STATUS.md + 小 sweep json（<100K）并入 experiments/ugnn/archive/（保留文件名前缀 expNN_）；exp12 的 sweep_raw_round{1,2}.json(325M) 不留
- lib/ 迁移：train_capacity.py、train_centernet.py、eval_scale.py、eval_pipeline.py、train_unet.py、eval_watershed.py → experiments/ugnn/lib/，并修 6 处 sys.path/import（eval_centernet.py、train_band8.py 及 archive 后仍存活的脚本）；exp20/gt_records 链接若指向 exp17 则保持（exp17/gt_records 保留）

### 验证（执行后必做）
- 100 图 fast eval 门：与 canonical RLE CRC 逐位一致（参照 exp13 determinism_crc 模式）
- train_band8.py 50-step 冒烟（GPU 空闲后）
- pytest tests/（保留子集）全过；ruff 全仓
- 依赖测绘时的坑：eval_centernet 的 --arch e10 strict-load train_capacity.SeedNet；baselines16m PYTHONPATH 含 src
