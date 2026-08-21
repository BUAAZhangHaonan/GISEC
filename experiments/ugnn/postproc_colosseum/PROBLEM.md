# Post-processing Colosseum (Round 2): E9 CPU watershed post-processing speedrun

Repo: $R = /home/k100/zhn/electronic-components-grasp-and-segment/gisec
Arena: $R/experiments/ugnn/postproc_colosseum/
Model: E9 CenterNet-seeds (exp09, smp UNet resnet18 three-head, 14.4M).
GPU forward is 8.1 ms/img; the full eval pipeline is ~470 ms/img wall.
The entire gap is CPU post-processing. Your job: make it fast without
losing correctness. This round only the FINAL ("centernet") hot path
is in scope. The oracle_gt_centers config and the 100x scene bootstrap
are diagnostic and NOT in scope.

## 1. The task

Given, per image, the model outputs and the calibrated depth, produce
the COCO RLE instance result list, faster than the reference
implementation, with the correctness gates of section 5 intact.

## 2. Data flow (reference hot path, per image)

Inputs (per image):
- sem: uint8 [1024,1024], sigmoid(sem_logit) > 0.5 foreground mask
- hm: float32 [256,256], sigmoid center heatmap (stride 4 head)
- off: float32 [2,256,256], regressed sub-cell offsets (dy,dx)
- depth: float32 [1024,1024], calibrated metric depth (input-only)
- h, w, image_id

Steps (reference/postproc_ref.py, byte-identical to
exp09_centernet_seeds/eval_centernet.py _cn_markers + _worker_one
FINAL branch):
1. cn_markers: 3x3 maximum_filter NMS on hm, threshold HM_THR=0.3,
   decode y = cell*4 + off, clip/round -> marker coords (~50/img)
2. elevation: sobel(depth) gradient magnitude (input-only!)
3. markers raster: paint coords into int32 label image
4. watershed: skimage watershed(elev, markers, mask=sem)
5. postprocess merge: regions < 32 px merged into the 4-neighbor
   adjacent region with the longest shared boundary
6. instance extract: per label, mask + area, drop area <= MIN_AREA=16
7. to_results: top-100 by area, score = area-normalized, COCO RLE

Note which steps are input-only (depth -> elevation; depth load):
these may be precomputed and cached. Anything depending on sem/hm/off
must be computed per image.

## 3. Data package

- data/dumps/<image_id>.npz: sem (u8), hm (f16), off (f16),
  depth (f32). 250 val images: 20 fewest-GT, 20 most-GT, 210 random
  (seed 42). hm/off are stored f16; decode to f32 before use — the
  reference outputs were produced from exactly these dumps.
- data/dumps/metajs.json: image_id/file_name/height/width.
- data/reference_outputs/reference_outputs.json: reference RLE
  results per image (the correctness ground truth).
- data/MANIFEST.md5: md5 of every dump + reference output.
- reference/postproc_ref.py: importable reference implementation;
  exposes run(image_id, sem, hm, off, depth, h, w) -> results list.
- bench/correctness.py, bench/timing.py: the gates (below).

## 4. Output format

list of COCO result dicts per image:
  {"image_id": int, "category_id": 1, "segmentation": RLE dict,
   "score": float}
RLE must be COCO-compressed (pycocotools encode) and deterministic.

## 5. Correctness gates (all must pass; judge's rerun is final)

  C1 instance count: |n_pred - n_ref| == 0 on >= 95% of the 250
     images AND max deviation <= 1 on every image.
  C2 matching: reference vs contestant instances matched greedily by
     best IoU; mean IoU >= 0.995.
  C3 probe AP: on the 50-image probe set (50 lowest image_ids of the
     package), COCO segm AP |contestant - reference| <= 0.01.
  C4 determinism: same input, two runs -> bitwise-identical RLE
     output.
  C5 cache discipline: any precomputed cache MUST be keyed by
     (split, image_id) and validated on val. A cache that collides
     across splits/ids (round-1 team_a: train/val ann-id collision
     silently returned train data for val) is instant
     disqualification. Caches that bake in model outputs keyed to
     these 250 dumps only (lookup-table cheating) are likewise
     banned: your entry must run on ANY val image.

## 6. Timing protocol

- Official timing: judge runs bench/timing.py serially —
  single process, warmup 10 images, 50 images, median ms/img.
- Contestant self-reported timing is advisory only.
- Throughput claim (secondary metric): your entry may additionally
  report imgs/s with <= 8 worker processes and <= 32 GB RAM.
- All heavy compute wrapped in systemd-run --user
  -p MemoryMax=32G -p CPUQuota=800%; <= 8 parallel processes;
  GPU: at most one card, and only for dump generation, not for the
  contestant hot path (this round is CPU post-processing).
  > **Revision 2026-08-21 (user directive, post-verdict):** the
  > GPU hot-path restriction is LIFTED — GPU acceleration is now
  > allowed and CPU is no longer mandated. Recorded after the
  > round-2 verdict; see ARENA.md §3.1 for the no-rerun review.
- Write the systemd unit names you used into your STATUS.md.

## 7. Baseline (judge-measured, 2026-08-21)

Reference single-process per-step medians (50 val imgs, warmup 10):

| step               | median ms/img |
|--------------------|---------------|
| depth load (npz)   | 1.8 |
| elevation (sobel)  | 67.8 |
| cn_markers         | 2.0 |
| markers raster     | 0.4 |
| watershed          | 86.5 |
| merge postprocess  | 9.9 |
| instance extract   | 48.6 |
| to_results (RLE)   | 290.2 |
| total              | ~507 |

Reference single-process end-to-end (dump->RLE, bench/timing.py):
see bench/BASELINE.md (filled by the judge after the sanity run).
Current production pipeline: Pool(6), 470 ms/img wall (incl. 8.1 ms
GPU forward). Biggest head: to_results RLE encoding (290 ms),
watershed (86 ms), elevation (68 ms), instance extract (49 ms).

## 8. Submission

Your directory: $R/experiments/ugnn/postproc_colosseum/team_<x>/
- postproc.py exposing run(image_id, sem, hm, off, depth, h, w)
- STATUS.md: unit names used, self-timed numbers, what you changed
- any precompute script you used, keyed per C5

Judging: correctness gates first (one-vote veto), then official
timing. Fastest median ms/img among fully-passing entries wins.
