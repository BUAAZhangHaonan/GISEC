# 2026-04-13 Query Alpha Official Baseline Summary

Prepared scaffold for the official small and medium baseline runs. No official training or evaluation has completed yet, so the metric fields below are intentionally left as `TBD`.

## Expected Official Layout

- official layout root: `output/experiments/2026-04-13-query-alpha-official`
- stable alias: `output/experiments/query_alpha_official`
- per-run tree: `{train,eval}/<variant>/...`

## Metrics Table

| variant name | segm/AP | bbox/AP | boundary/IoU | train wall time | comparison vs active RGB refine-only baseline |
|---|---|---|---|---|---|
| `query_small_resnet18` | `TBD` | `TBD` | `TBD` | `TBD` | pending official run |
| `query_medium_resnet34` | `TBD` | `TBD` | `TBD` | `TBD` | pending official run |

## Gate Evaluation

The baseline gate checks are written in plain language here:

- Does the official run finish cleanly for both baseline variants.
- Do the stored metrics support the documented query-alpha acceptance criteria.
- If the gate compares against the active RGB refine-only reference, is the result at least competitive with that baseline.

Status: pending because training and evaluation have not run yet, so there are no real metrics to compare.

## Recommendation

NO-GO: official small and medium baseline runs have not been executed yet.
