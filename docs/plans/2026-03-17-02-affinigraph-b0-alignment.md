 # AffiniGraph B0 Alignment Plan

 ## Goal
 Lock down the true baseline so comparisons stay honest and every variant is explicitly defined rather than driven by implicit substring logic.

 ## Scope
 - Introduce `affinigraph.config.variants.VariantSpec` to codify B0–G5 toggles.
 - Update loaders/graph builders so shape stats are used only when the spec asks for `use_shape_stats`.
 - Expand regression tests to capture B0 behaviour and enforce the new variant contracts.

 ## Key Changes
 - Replace the `VARIANT_FLAGS` table with declarative spec entries that expose `use_shape_stats`, `use_rgb_reference_similarity`, `use_depth_reference_similarity`, and `use_learned_edge_scorer`.
 - Amend `build_graph_batch` and `GraphRefiner` consumers to gate the sixth edge-feature channel behind the spec flag and to keep `shape_stats` away from B0.
 - Add explicit tests that fail if B0 still sees shape stats, while G2+ continuing to require them.

 ## Acceptance
 - B0 produces the same edge features as a pure `boundary+affinity` heuristic (shape-derived channel remains zeroed).
 - G2 introduces `shape_stats` for the first time, and G3–G5 only enable the RGB/D prior channels that their names imply.
 - Regression tests flag any drift in these semantics before experiments start.

 ## Verification
 - `pytest tests/test_variant_spec.py` passes and fails if spec flags deviate from expectations.
 - `build_graph_batch` output for B0, G2, G5 is consistent with their declared feature toggles.
 - Documentation or release notes call out the new variant API so researchers know the exact definitions.
