# 2026-04-13 Query Alpha Deferred Variants Summary

Prepared scaffold for the deferred reference-conditioned, graph-integrated, and reference-plus-graph query variants. No official training or evaluation has completed yet, so the metric fields below are intentionally left as `TBD`.

## Expected Official Layout

- official layout root: `output/experiments/2026-04-13-query-alpha-official`
- stable alias: `output/experiments/query_alpha_official`
- per-run tree: `{train,eval}/<variant>/...`

## Metrics Table

| variant name | segm/AP | bbox/AP | boundary/IoU | train wall time | comparison vs active RGB refine-only baseline |
|---|---|---|---|---|---|
| `query_ref_resnet18` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD against the active RGB refine-only baseline (~0.576 segm/AP)` |
| `query_ref_resnet34` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD against the active RGB refine-only baseline (~0.576 segm/AP)` |
| `query_graph_resnet18` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD against the active RGB refine-only baseline (~0.576 segm/AP)` |
| `query_graph_resnet34` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD against the active RGB refine-only baseline (~0.576 segm/AP)` |
| `query_refgraph_resnet18` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD against the active RGB refine-only baseline (~0.576 segm/AP)` |
| `query_refgraph_resnet34` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD against the active RGB refine-only baseline (~0.576 segm/AP)` |

## Gate Evaluation

| Criterion | Met? | Supporting metric values |
|---|---|---|
| Reference-conditioned variants are allowed only after the base query line is stable and interpretable. | `TBD` | `TBD` |
| Graph-integrated variants are allowed only after the base query line is stable and interpretable. | `TBD` | `TBD` |
| Combined reference-plus-graph variants are allowed only after both the reference and graph branches have shown separate value. | `TBD` | `TBD` |

## Recommendation

NO-GO: deferred query variants have not been executed yet.
