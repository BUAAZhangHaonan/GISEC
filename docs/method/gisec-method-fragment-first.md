# GISEC Method Method Design

## 1. Task Definition and v1 Limits
`GISEC` targets RGB-D instance segmentation for densely cluttered electronic components. The hard part is usually not deciding whether a region is an electronic component. The hard part is deciding whether several visually broken pieces still belong to the same instance when color, highlight, occlusion, narrow pins, or depth noise split that instance into multiple fragments.

`GISEC v1` is a valid Stage 1 baseline, but it is capped by five design limits:
- it supervises local pixel adjacency instead of true instance ownership,
- it averages all prototype views into one blurred reference,
- it builds a graph mostly from touching fragments,
- it merges fragments with plain threshold union,
- it uses depth only as a weak conditioning hint and tolerates noisy graph labels.

For that reason, `GISEC Method` is defined as a replacement method, not a small patch on top of `v1`.

## 2. Overall Framework and Data Flow
`GISEC Method` follows a supervision-first pipeline:

1. the query RGB image and depth-derived geometry maps go through a lightweight U-Net-style backbone,
2. the backbone predicts foreground, boundary, and pixel-to-core ownership offsets,
3. a prototype router selects and mixes a small set of reference prototypes instead of one averaged prototype,
4. the predicted fragments become graph nodes,
5. the graph builder creates both contact edges and bridge edges,
6. a lightweight graph scorer predicts same-instance likelihood for each edge,
7. a constrained merge module greedily accepts only those merges that remain structurally plausible under reference and geometry checks.

The design target is simple: let the pixel stage make mistakes that are easy to repair later, then make the graph stage repair them without causing catastrophic false merges.

## 3. Module Definitions
### `DepthGeometryStem`
This module is always active. It converts raw depth into three cheap geometry maps:
- normalized depth,
- depth gradient magnitude,
- depth discontinuity edge.

These maps enter the backbone from the start. The purpose is to expose geometric events directly, especially places where RGB looks continuous but geometry says the surface is broken.

### `PrototypeRouter`
This module replaces the `v1` single averaged prototype. Each part first samples a compact reference pack with `pose_farthest` coverage; the current default target is `16` views for full runs, while smoke runs can override this downward. From that pack, the model keeps `K=6` prototype slots. The query first produces a global descriptor, then uses top-2 soft routing to mix the most relevant prototype slots. If camera pose metadata exists, it can act as a routing prior that suppresses obviously incompatible views, but it does not become a backbone feature.

This keeps reference conditioning lightweight while preserving view-specific structure that was lost in `v1`.

### `OwnershipHead`
This head keeps the same `2-channel` output width as the old affinity branch, but the meaning changes completely. Instead of predicting whether the right or lower neighbor belongs to the same instance, each foreground pixel predicts a 2D offset to the centroid of the largest eroded core component of its instance.

This makes the prediction semantic closer to instance ownership. Two fragments do not need to touch each other visually if both still point to the same underlying core.

### `GraphBuilderV2`
This module converts fragments into a sparse candidate graph with two edge types:
- `contact edges` for fragments that meet across a boundary scan,
- `bridge edges` for fragments that are close but separated by a short corridor with weak boundary and weak depth discontinuity.

Each fragment keeps at most `top-3` bridge candidates. The graph is therefore recall-aware without becoming dense and expensive.

### `GraphEdgeScorerV2`
The graph scorer remains lightweight on purpose. The main improvement is not a larger GNN. The main improvement is that node and edge features become more meaningful:
- ownership consistency,
- depth discontinuity cues,
- reference-route similarity,
- edge type,
- fragment geometry,
- optional shape-quantile compatibility.

### `ConstrainedGreedyMerge`
This module replaces threshold union-find. Edge candidates are sorted by score from high to low. A merge is accepted only if the merged component stays plausible under several guard rails:
- merged area ratio remains inside reference `q10-q90`,
- merged aspect ratio remains inside reference `q10-q90`,
- the corridor does not cross a strong depth discontinuity,
- ownership landing points do not clearly diverge.

This is intentionally conservative. In this task, over-splitting is usually repairable later. Over-merging often ruins the instance outright.

## 4. Supervision Signals
### Foreground
- target: binary foreground mask
- loss: `BCE + Dice`

### Boundary
- target: boundary band from instance masks
- loss: `balanced BCE`

### Ownership Offset
- target: 2D offset from each foreground pixel to its GT core centroid
- valid region: foreground only
- loss: `Smooth L1`

### Graph Edge
- target: same-instance label for graph edges after purity filtering
- loss: `balanced BCE`
- training note: add hard-negative emphasis so visually tempting but incorrect edges matter more

## 5. Training Objective and Sample Filtering
The total objective is:

`loss_total = loss_fg + loss_boundary + lambda_offset * loss_ownership + lambda_graph * loss_graph`

The initial bias is to keep `lambda_offset` and `lambda_graph` moderate. The backbone should first learn to produce recoverable fragments before graph terms dominate.

Graph supervision is filtered in two stages.

First, node purity is computed for every fragment:
- `purity = dominant_instance_pixels / fragment_pixels`
- low-purity fragments are ignored for graph supervision

Second, edge purity is checked:
- if the contact band or bridge corridor mixes multiple GT instances, that edge is ignored

This removes a major source of label noise from `v1`, where a fragment spanning multiple GT instances could still be assigned one hard majority label and then contaminate graph learning.

## 6. Inference and Constrained Merge
Inference runs in the following order:

1. predict `fg_logits`, `boundary_logits`, `ownership_offsets`, and `feature_map`,
2. derive fragments from foreground and boundary maps,
3. estimate node geometry and ownership landing behavior,
4. build contact edges and bridge edges,
5. score all edges with the lightweight graph scorer,
6. attempt merges in descending score order,
7. reject merges that violate reference quantiles, depth continuity, or ownership consistency,
8. export the final merged instances with the existing artifact contract.

The critical policy is explicit:
- a doubtful merge should be rejected,
- the merge module should prefer leaving two fragments separate over collapsing two real instances into one.

## 7. Ablation Matrix
- `A0`: current `v1` best carry-over baseline, default record `G4 fixed-eval`
- `A1`: `A0 + ownership offset`
- `A2`: `A1 + purity-filtered graph supervision`
- `A3`: `A2 + multi-prototype routing`
- `A4`: `A3 + always-on depth geometry`
- `A5`: `A4 + contact + bridge graph builder`
- `A6`: `A5 + constrained greedy merge`
- `S1`: `A6 + pose-aware prototype routing prior`

Interpretation rules:
- `A1` must prove that ownership is better than local adjacency for same-instance grouping,
- `A5` must prove that graph gain comes from better candidate recall,
- `A6` must prove that safer merge rules reduce catastrophic chain unions,
- `A3/A4` must be explainable as separate reference and geometry gains.

## 8. Minimal Viable Experiment Order
Current stage is documentation-first and GPU-gated. Only the following actions are allowed now:
- code audit,
- method docs,
- interface docs,
- `pytest`,
- CLI dry runs,
- offline inspection of existing logs and outputs.

When GPU capacity is available again, the minimal experiment order is:

1. `E0`: freeze `A0` protocol and logging contract,
2. `E1`: short-run `A1`,
3. `E2`: short-run `A2`,
4. `E3`: short-run `A5`,
5. `E4`: short-run `A6`,
6. `E5`: short-run `A3 + A4`,
7. only then consider full `0831_1K / 1024 / 20 epochs`.

Full-matrix training is blocked unless short runs show that the new mainline beats `A0` and that the improvement is interpretable. If those gates fail, the correct response is to revise the method, not to spend more GPU on a broken design.
