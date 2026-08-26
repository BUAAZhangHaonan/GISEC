# E5: Route Verdict — U-Net + GNN (1566 val)

Date: 2026-08-18. Evidence: E1-E4 RESULT.md in this directory. No new
experiments were run for this verdict.

## 1. Verdict

The original U-Net+GNN conception — a dense semantic mask plus a graph
network that merges fragment components back into parts — is dead, and it
died for a reason none of us guessed at design time: the merge direction
has no input. E1 showed the identity signal exists (pair AUC 0.991) but
the never-merge baseline scores 0.9916, so a merge rule can only lose;
E2 then showed that even when fragments do exist, a wrong merge (0.3559)
is worse than no merge (0.4083); E3 closed the loop with the real
mechanism — a dense segmenter produces a *union* mask where 91% of parts
are already fused (864 CCs vs 9494 GT instances), so there are no
fragments to merge at all. The actual problem is the opposite one:
splitting fused blobs. E4 showed that splitting is a depth problem, not
a graph problem — a zero-training depth-gradient watershed lifts 0.0287
to 0.3125 where an RGB gradient collapses to 0.0260. **"GNN" is hereby
removed from the route name.** What survives is a depth-first
small-model instance pipeline; whether it earns another training run is
decided in section 4.

## 2. Surviving Assets

- Depth is the strongest structural signal in this task. E1: pair AUC
  0.991 (depth+spatial), 26.4x between/within group variance. E4: depth
  gradient elevation 0.3125 vs RGB gradient 0.0260 — a 12x gap; with GT
  semantics, depth watershed alone lifts 0.0287 (CC) to 0.4933.
- Small-model semantic segmentation is solved. E3: mIoU 0.945 at 14.5M
  params; oracle-semantic control shows semantics account for only
  +0.010 of the AP gap — instance recovery is 97% of the problem.
- Instance scoring is a non-problem. E2: every scheme including constant
  0.5 saturates segm AP at 0.9901 (1.00x oracle); noise sigma 0.3 costs
  nothing. No scorer head is ever needed.
- Training efficiency is proven. E3: 20 epochs in 35.7 min on one GPU,
  mIoU still rising at epoch 19. A retrain with an extra head is ~1h,
  not a day.

## 3. Gap Attribution: 0.3125 -> 0.5381 (M2F swin-t)

| factor | evidence | est. gain |
|---|---|---|
| Seed placement (not seed count) | E4-b: GT seed *count* via greedy top-N peaks *hurts* (0.1798 < 0.3125); placement is the hard half | largest |
| Depth boundary precision | E4-a: GT semantics cap depth-watershed at 0.4933; AP50 0.6198 vs AP75 0.2787 — boundaries land in IoU 0.5-0.75 band | medium |
| Over-segmentation 1.5x | 96.8 pieces/img vs 63.7 GT; fusion 91% -> 8.2% undersplit but count still 1.5x | medium |
| Grid not saturated | AP still rising at md15 grid edge; 0.3125 is a lower bound for the operator family | small |

The 0.4933 GT-semantic ceiling says depth plateau/gradient flooding
alone cannot reach 0.99 — adjacent same-depth parts and soft depth
edges leak. Closing to 0.5+ needs learned seeds and/or boundary
refinement, i.e. training.

## 4. Go / No-Go

**Continue — one conditional experiment, then re-judge.**

Reasons:
1. The route's strategic premise (small model, depth-native, cheaper
   than M2F) is intact and now has a working zero-training split
   operator at 0.3125 with headroom left at the grid edge.
2. The single binding constraint is seed placement, and it has a cheap
   learned fix: a center/keypoint heatmap head on the existing E3
   U-Net. This is exactly the fix E3's conclusion prescribed, and E4-b
   proved the failure mode is placement, not count.
3. The downside is bounded: ~1h retrain, 14.5M params.

**Conditional experiment (E6):** add a center heatmap head to the E3
U-Net (2-channel output: semantic + center; ~+0.1M params), retrain 20
epochs (~1h), seed the depth-gradient watershed with predicted centers
instead of depth peaks, keep md sweep {9, 15, 21}.

**Pass line: segm AP >= 0.42** on 1566 val (scene bootstrap CI lower
bound >= 0.38). Rationale: 0.42 is ~80% of M2F's 0.5381 with 1/3 the
parameters, and it clears the E4 gray zone decisively — a learned seed
must at least beat the 0.4933 GT-semantic ceiling trend line, otherwise
learning is not paying for itself.
- Pass -> proceed to the roadmap in section 5.
- Fail (< 0.42) -> route closed; return to the mainline. No second
  rescue attempt; boundary refinement is only funded after 0.42.

## 5. Scale-up Roadmap (only if E6 passes)

1. **1566 consolidation.** md sweep to convergence; fix over-split 1.5x
   (merge small same-depth neighbors). Pass: AP >= 0.45 on 1566 val.
2. **32254 transfer.** Same recipe, no architectural change. Reference:
   concat-fusion ceiling model 90.63 @ 32254; M2F swin-t 0.5381 @ 1566.
   Pass: within 0.05 of the M2F-family number on the large val split.
3. **Efficiency Pareto, not AP-max.** The route never out-builds a good
   M2F recipe (90.63) on raw AP. Its claim is: same AP at ~1/4 params
   and faster inference. Report AP / params / FPS jointly; if at any
   step the Pareto claim fails, stop there.

## 6. Method Lessons

1. Zero-training gating works: four days of E1/E2/E4 sims and controls
   killed and rebuilt a route with exactly one 35-minute training run
   (E3) burned.
2. Always compute the trivial baseline: E1's depth-rule "accuracy
   0.9776" was *worse than doing nothing* (never-merge 0.9916). A pass
   bar without a do-nothing control is not a pass bar.
3. Simulate the real failure mode, not the imagined one: E2 modeled
   fragment-merging, but E3 showed a dense segmenter *fuses* parts —
   the pipeline had the opposite problem. One real checkpoint exposed
   in 35 min what the GT-based sim could not.
