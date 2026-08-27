# decode_fix STATUS

- 2026-08-27 代码修复落地（C1/M5/m2/m4），ruff 过，pytest 16/16。
- sweep：unit gisec-decfix-sweep 完成（三变体×7 thr×500 图 + 对齐门 + seed）。
- 全量复现门/赢家全量：unit gisec-decfix-legacyfull 完成，0.8487991 ≈ 0.84880 PASS。
- 四线全过，canonical 维持；判决与产物见 RESULT.md。收尾 commit+push。
- 2026-08-27 C2/M3 stat 修复：lib/scene_boot multiplicity-aware bootstrap（三验证门 + 全量 G1 ≤1.1e-16 + prereg 0.8487991 复现全过）；E20 canonical CI 重算 segm CI95 [0.83217,0.86454]（2000 draws，旧 [0.83678,0.86363] 作废）；fixed-vs-legacy 全量配对 CI −0.00187 [−0.00354,−0.00039] 不含 0，legacy 维持；M3 cross-fit 判决平局（crossfit_decode.json）；E17 无 ckpt 不可补配对 CI。
