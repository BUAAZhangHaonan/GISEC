This directory only holds thin wrappers reused across E1-E5: pair-feature
building, the scoring function, and dataset wrappers. All data loading, COCO
evaluation, union-find, and connected-component logic must be imported from
the `gisec` package directly (model/targets/data/train/decode/postproc/eval
all live in `src/gisec/` since the 2026-09-02 consolidation); copying
implementations here is forbidden.
