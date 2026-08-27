# decode_fix STATUS

- 2026-08-27 代码修复落地（C1/M5/m2/m4），ruff 过，pytest 16/16。
- sweep：unit gisec-decfix-sweep 完成（三变体×7 thr×500 图 + 对齐门 + seed）。
- 全量复现门/赢家全量：unit gisec-decfix-legacyfull 完成，0.8487991 ≈ 0.84880 PASS。
- 四线全过，canonical 维持；判决与产物见 RESULT.md。收尾 commit+push。
