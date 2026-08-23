# gisec-e13-fullboot2

- unit: gisec-e13-fullboot2.service (systemd --user); 首次尝试 gisec-e13-fullboot 用 64G MemoryMax 被 systemd-oomd 击杀（GT 加载阶段），重启为 160G cap
- started: 2026-08-22 23:31 (retry of 23:29 fullboot)
- cmd: systemd-run --user --unit=gisec-e13-fullboot2 -p MemoryMax=160G -p CPUQuota=3200% --working-directory=experiments/ugnn/exp09_centernet_seeds /home/k100/miniconda3/envs/gisec/bin/python eval_centernet.py --arch e10 --profile full --ckpt ../exp10_semantic_capacity/runs/best.pth --out ../exp13_integrate/eval_report_full_20260822.json
- scope: full 3276 val, FINAL + oracle + seed + GT stats（新 canonical 0.82137 的 scene bootstrap CI + oracle 探针）
- ETA: ~1.5-2h；启动确认 100/3276 @ 0.37 s/img, rss 9.5G, active (running)
- follow: journalctl --user -u gisec-e13-fullboot2 -f
- output: experiments/ugnn/exp13_integrate/eval_report_full_20260822.json
- 注: huggingface.co HEAD 重试告警为代理自签证书所致，模型权重回退本地缓存加载成功，无影响
- 单元 gisec-e13-fullboot2 已于 08-23 00:28 完成
