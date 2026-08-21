# E10 RESULT: semantic capacity recovery (three-head, widened)

## Preregistered pass lines (frozen before results)

- P1 (main): oracle segm AP >= 0.79 (recover the E8 two-head level;
  E9 oracle was 0.7359, E8 was 0.7952)
- P2: seed median < 8 px AND <8px rate >= 90% (E9: 2.35 px / 96.3%;
  seeds must not be sacrificed)
- P3: FINAL segm AP >= 0.75 (E9 FINAL 0.7254)
- P4: total params <= 19M (stay under 40% of the 47.4M M2F)

Gray zone: oracle in [0.76, 0.79) -> gray, next step is ablation
(isolate decoder-width vs sem-weight). P2 broken -> FAIL regardless
of other lines.

## Design rationale

E9 located the gap: adding the seed head dropped val mIoU
0.9989 -> 0.9968 and oracle 0.7952 -> 0.7359 while FINAL reached
98.6% of oracle — seeds are solved, semantics is the binding
constraint. Mechanism: seed-head (focal + offset) gradients flow
through the shared 3.15M decoder and starve the semantic path.

Three coupled changes, all attacking that mechanism; everything
else (heads topology, recipe: AdamW 3e-4 cosine, 20 epochs from
scratch, batch 8@1024, 16-worker gt_records loader, aug) is
replicated from E9:

1. Decoder widen (256,128,64,32,16) -> (384,192,96,48,24):
   3.15M -> 5.61M params. Capacity for BOTH tasks; the seed head
   keeps its structure (only the first conv widens 16 -> 24 in).
2. Semantic head deepened: the 145-param single conv becomes a
   3-layer block (24->24->12->1, ~8.0K params). Without this the
   widened decoder bottlenecks at a 1-conv decision.
3. SEM_W = 2 (was 1): restores the semantic gradient share against
   the two seed terms (focal and offset weights unchanged at 1).

Measured params (printed at train start, asserted vs budget):
encoder 11.180M + decoder 5.610M + seg head 7,993 + seed head
52,803 = 16.851M total (E9: 14.38M; +2.47M, budget 19M).

## Numbers

Canonical run: ugnn-e10-eval2 (systemd --user, MemoryMax=64G, CPUQuota=3200%),
2026-08-22, `eval_centernet.py --arch e10 --ckpt runs/best.pth`, full val 3276
imgs, report at runs/eval_report.json (bit-identical rerun of the 08-22 05:07
journal run that crashed at the final JSON write; all metrics reproduced
digit-for-digit).

| line | segm AP | segm AP50 | segm AP75 | bbox AP | n_pred/img |
|---|---|---|---|---|---|
| FINAL (centernet seeds) | 0.7697 | 0.8656 | 0.7787 | 0.6890 | 50.77 |
| oracle (GT centers)     | 0.7734 | 0.8750 | 0.7798 | 0.6928 | 51.10 |

- Bootstrap (210 scenes x 100): segm 0.7697, CI95 [0.7529, 0.7904];
  bbox 0.6897 [0.6703, 0.7131].
- Seed precision: median 2.30 px, p90 4.92 px, <8px 96.36% (E9: 2.35 px).
- FINAL is 99.4% of oracle (E9: 98.6%) — seed placement remains solved.
- Train best val mIoU 0.9984 (ep18; E9 best 0.9968 -> the widened decoder +
  SEM_W=2 did recover the semantic head's fit).
- Undersplit piece rate 7.03% FINAL vs 6.63% oracle; oversplit ~0.001%.
- Wall 0.30 s/img (E9 postproc cache + RGB decode cache path).

## Verdict

| line | target | got | verdict |
|---|---|---|---|
| P1 oracle segm AP | >= 0.79 | 0.7734 | FAIL (gray zone [0.76, 0.79)) |
| P2 seed median / <8px | < 8 px / >= 90% | 2.30 px / 96.36% | PASS |
| P3 FINAL segm AP | >= 0.75 | 0.7697 | PASS |
| P4 params | <= 19M | 16.85M | PASS |

vs E9 (same heads, narrow decoder): oracle 0.7359 -> 0.7734 (+3.75 pt),
FINAL 0.7254 -> 0.7697 (+4.43 pt), mIoU 0.9968 -> 0.9984, seeds unchanged.
The semantic-starvation diagnosis was correct: widening + SEM_W=2 bought back
most of the third-head cost.

Gray-zone conclusion (preregistered): oracle 0.7734 in [0.76, 0.79) -> not a
clean fail, next step is ablation, not a new design. E8 two-head oracle 0.7952
is NOT fully recovered; residual 2.2 pt. Three hypotheses to ablate:
1. offset-head interference (focal+offset grads still compete with semantics
   even in the widened decoder — try dropping the offset head or detaching);
2. capacity still insufficient (5.61M decoder vs 47.4M M2F — try wider still
   or encoder-side capacity);
3. training length (20 ep from scratch; E8 lineage was a warm-start resume —
   longer schedule may close the rest).

Recall-gap observation: n_pred 50.77/img vs GT 55.46/img (heatmap emits 54.71
markers/img) — ~4.7 objects/img never get a seed, so oracle itself is
recall-capped; the seed-precision metrics hide a coverage deficit. Raising
marker coverage (lower heatmap threshold / top-k decode) is an orthogonal,
zero-training lever on the oracle path.
