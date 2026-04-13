# GISEC Experiments

This directory collects experiment-facing artifacts derived from the Stage 1 matrix.

## Intended Contents

- per-suite summaries generated from `output/experiments/*`
- extended metrics tables for `B0/G1/G2/G3/G4/G5`
- notes for reruns, regressions, and negative-result branches

## Standard Workflow

After a suite finishes, generate the textual reports:

```bash
python scripts/analysis/summarize_suite.py \
  --suite-root output/experiments/gisec_0831_matrix \
  --output-json docs/experiments/gisec_0831_matrix_summary.json \
  --output-md docs/experiments/gisec_0831_matrix_summary.md
```

```bash
python scripts/analysis/write_extended_metrics_table.py \
  --suite-root output/experiments/gisec_0831_matrix \
  --output docs/experiments/gisec_0831_matrix_extended_metrics.md
```

Commit generated summaries alongside the code or config changes that produced them.
