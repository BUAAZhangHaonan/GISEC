# GISEC Query Alpha Metrics Surface

All `query-alpha` experiments must report the same common surface.

## Headline Metrics

- `segm/AP`
- `bbox/AP`

## Common Diagnostics

- `pred_count_mean`
- `gt_count_mean`
- `best_mask_iou_mean`
- `best_bbox_iou_mean`
- `failure_summary`

## Query-Only Phase Diagnostics

- `object_count_mean`
- `split_count_mean`
- `avg_cores_per_object_mean`
- `pred_fg_rate_mean`
- `pred_boundary_rate_mean`

## Later Rescue Diagnostics

- `reference_routing_summary`
- `graph_rescue_summary`

The rule is simple: no module is allowed to invent a private evaluation surface. Every stage must remain readable against the same shared metrics and diagnostics.
