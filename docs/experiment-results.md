# GISEC Experiment Results

This page keeps the official result ladder for the standalone GISEC package.

## Best Official RGB Ladder

| Stage | segm/AP | bbox/AP | boundary/IoU | train wall time (s) | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `base_rgb_1024` | `0.5495995386078752` | `0.5140047832669681` | `0.19392113294585225` | `32775` | Stage-1 baseline |
| `base_rgb_1024_refine` | `0.5761366653940664` | `0.5155950306627068` | `0.25118819472440657` | `50120` | Best stored official result |
| `base_rgb_1024_refine_ref` | `0.5747495053887953` | `0.5141577902090311` | `0.25009854065035914` | `55372` | No improvement over refine |
| `base_rgb_1024_refine_ref_graph` | `0.5745757912158308` | `0.5153391542700804` | `0.24883020894828495` | `75035` | No improvement over refine |

## Interpretation

The official ladder says the RGB refine stage is the best stored result. The extra reference and graph stages are useful parts of the architecture, but the stored metrics do not beat the refine-only stage.

The result above comes from:

- `configs/model/base_rgb_1024_refine.yaml`
- `configs/data/ecc_20260318_1k_1566.yaml`

The main takeaway is simple: the current standalone GISEC package is anchored by the refine stage, and the later rescue stages stay available for the harder cases without changing that published best.
