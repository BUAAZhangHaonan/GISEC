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

(filled after training + eval_centernet.py run)

## Verdict

(pending)
