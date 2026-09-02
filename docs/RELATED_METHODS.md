# Related methods: mechanism map & official-implementation status

Compiled 2026-09-03 from full-text reading (Panoptic-DeepLab via MCP), arXiv
abstracts (UOIS-Net, DWT), the official code we actually run, and the web
survey of official implementations. Companion to `docs/BASELINE_ATLAS.md`
(results/where); this file records *how each method works* and *how it maps
onto GISEC*.

## Mechanism comparison table

| Method | Instance representation | Center/offset supervision | Input | Grouping at inference | Official code | Our 32254 run |
|---|---|---|---|---|---|---|
| **GISEC** (ours) | semantic mask + CenterNet heatmap (stride 4) + offset | focal on size-adaptive Gaussian; anchor = discrete support-constrained p* | RGB-D | rank-fused (depth+sem-logit gradient) watershed, bucket queue | — | E25 **0.87350** (canonical) |
| Panoptic-DeepLab (CVPR20) | semantic + center heatmap + per-pixel offset | **MSE on 2-D Gaussian (σ=8px), λ=200**; offset L1 λ=0.25 | RGB | max-pool NMS (k=7, thr 0.1, top-200) → each fg pixel joins nearest center after offset shift; instance score = objectness × class score | bowenc0221/panoptic-deeplab (d2) | R50 64K@bs8 running (6401 GPU1) |
| Deep Watershed Transform (CVPR17) | learned energy map; instances = basins | end-to-end energy regression (successive refinement) | RGB | **single-energy-level cut → connected components** (not flooding) | min2209/dwt: TF1 **inference only, no training code** | not run (no trainable official impl) |
| UOIS-Net (ICRA20/RA-L) | depth-only center votes (2D/3D) → rough masks; RGB refinement | DSN votes + fg; RRN mask refinement | RGB-D (DSN on xyz, RRN on RGB crop) | 3D mean-shift clustering of votes (BlurringMeanShift) + patch RRN | chrisdxie/uois + TOD weights | **zero-shot done: AP 0.0003** (500 imgs; renderer-domain gap) |
| Fast UOIS (Actuators 2024) | center offsets → local extrema → adaptive mean-shift seeds | offset prediction | RGB-D lineage (UCN family) | adaptive clustering seed selection | **no official code found** (closed-source) | not run; lineage covered by UOIS-Net/UCN |
| UCN (ICRA21) | RGB-D metric embedding | cosine embedding loss | RGB-D | mean-shift in embedding space | NVlabs/UnseenObjectClustering + OCID ckpt | OCID-pretrained fine-tune queued (4029) |
| CellPose (Nat. Meth.) | cellprob + 2-ch flow field | flow MSE + cellprob BCE | RGB | step-following along flow field to cell center | official lib 3.1.1.1 | 20ep full-data running (4029) |
| StarDist (CVPRW18) | star-convex polygon: radial distances + probabilities | radial-distance + object prob regression | RGB | NMS over star-convex polygons | official lib 0.9.2 (TF) | 20ep full-data running (4029) |
| YOLOv8-seg | anchor-free mask prototypes + coefficients | task-aligned assigner, COCO pretrain | RGB | NMS on mask coefficients | official ultralytics 8.4.x | 20ep running (4029) |
| Mask R-CNN / Mask2Former | proposals / queries | box+mask / query matching | RGB | top-N proposals / per-query masks | torchvision / HF | done: 0.6638 / 0.4305 |

## What GISEC shares with, and differs from, each line

- **Panoptic-DeepLab is the closest structural relative** (semantic +
  class-agnostic center + offset). Differences that GISEC isolates against it:
  (1) support-constrained discrete anchor p* vs PDL's center-of-mass Gaussian
  (GISEC E24 showed 29.3% of stride-4 peak cells move under p*, 48.6% of small
  instances have out-of-mask centroids); (2) rank-fused depth+semantic-gradient
  elevation watershed vs PDL's offset-only nearest-center assignment; (3)
  CenterNet penalty-reduced focal vs PDL's MSE-on-Gaussian. Same budget, same
  data, official code — the cleanest "is the GISEC-specific machinery worth
  it" control.
- **DWT is the ancestor of the watershed decode** but its inference is a
  single-level cut of a *learned* energy, while GISEC's elevation is
  *constructed* (rank of sobel(depth) + 2·rank(sobel(sem logit))) with a
  deterministic bucket-queue flood; GISEC learns only the components
  (semantics, seeds), not the energy itself. No official training code
  exists, so it is reported, not run.
- **UOIS-Net / Fast UOIS / UCN are the RGB-D unseen-object line** most
  aligned with GISEC's "depth carries the task" thesis (they seed from depth).
  Their zero-shot failure on our renderer (AP 0.0003) plus the from-scratch
  UCN 1.03 on 1566 delimit how far tabletop priors transfer; UCN fine-tune
  will give the trained-in-domain number for this line.
- **CellPose / StarDist are the bio-imaging instance families**: center/flow
  and star-convex representations that assume near-convex, well-separated
  objects — exactly the prior electronic-component contact breaks (the E2/E15
  diagnosis). Their 32254 numbers quantify that prior's cost in our domain.

## Positioning statement (for the paper)

GISEC's contribution is **task-structure driven, not a new general
architecture**: (1) discrete support-constrained anchors for multi-connected,
non-convex components; (2) rank fusion to put depth gradients and semantic
logit gradients on one scale; (3) reliable geometry used directly in a
deterministic instance split; (4) small learned modules replacing query
matching in low-overlap, single-class, dense scenes; (5) systematic error
attribution (E15 forensics) for why the decomposition fits the task.

## Provenance notes

- Panoptic-DeepLab: full text read 2026-09-03 (MCP); losses/hyperparameters
  above are from the paper's Sec. 3 (λ_heatmap=200, λ_offset=0.25 in paper
  text; the d2 configs use 200 / 0.01 — we follow the released d2 configs).
- DWT: min2209/dwt repo states "Will be available soon" for training code
  (never released; last push 2018-12); community PyTorch ports exist but are
  unofficial.
- UOIS-Net: arXiv 2007.08073 is the journal version of 1907.13236; the
  experts' citation "[2] Fast UOIS → 2007.08073" conflated the two — the
  real Fast UOIS is Fu et al., Actuators 2024, 13(8):305 (closed-source).
- Official-implementation survey (web, 2026-09-02): Panoptic-DeepLab
  detectron2 build deprecated-but-functional; cellpose/stardist actively
  maintained; UCN/UOIS-Net frozen but complete with pretrained weights.
