# 2026-04-12 Active RGB Official Ladder Summary

This note exists so the report can survive cleanup of large raw artifacts under `output/experiments/`.

The RGB-first framing here matches the repo's current public surface in `README.md` and `docs/results/README.md`: the active line is RGB-first, and later RGB-D / rescue stages are follow-up questions rather than the front-door story.

## Official RGB Ladder

| Stage | segm/AP | bbox/AP | boundary/IoU | train wall_time_sec | Notes |
|---|---:|---:|---:|---:|---|
| `base_rgb_1024` | 0.5495995386078752 | 0.5140047832669681 | 0.19392113294585225 | 32775 | Stage-1 baseline. |
| `base_rgb_1024_refine` | 0.5761366653940664 | 0.5155950306627068 | 0.25118819472440657 | 50120 | Refine-only is best. |
| `base_rgb_1024_refine_ref` | 0.5747495053887953 | 0.5141577902090311 | 0.25009854065035914 | 55372 | Ref and ref_graph do not improve over refine. |
| `base_rgb_1024_refine_ref_graph` | 0.5745757912158308 | 0.5153391542700804 | 0.24883020894828495 | 75035 | Ref and ref_graph do not improve over refine. |

## Conclusion

`base_rgb_1024_refine` is the best stored official RGB stage.

