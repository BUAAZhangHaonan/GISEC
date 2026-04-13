# Legacy Owner-Union Graph-Merge Analysis Note

## Context
- Prior learned result: `segm/AP = 0.420` from the existing learned owner-union summary.
- Oracle upper bound: `segm/AP = 0.849` from the existing oracle summary.
- The prior learned checkpoint is not available locally in this workspace, so pre-rerun runtime inference on that checkpoint is blocked for now.

## Diagnostic evidence already available
- Existing recovery notes point to collapse-by-over-merging rather than a pure mask-encoding failure.
- The current evidence set shows too few predicted instances, many wrong extents, and merge behavior that is still too permissive.
- The merge path is the narrowest credible place to intervene without rewriting the graph pipeline.

## Selected improvement
- Chosen improvement: stronger graph merge constraints.
- Concrete patch: add an area-balance guard in `_merge_allowed()` so merges are rejected when `min(area_ratio) / max(area_ratio) < 0.25`.

## Expected impact
- This should cut the obvious over-merges first.
- It should keep the change narrow and preserve the existing graph pipeline shape.
- It may reduce collapse into a single large blob while leaving the rest of the legacy training and evaluation flow untouched.

## Rerun status
- Official rerun is still pending because the original learned checkpoint is not present locally.
- When the checkpoint becomes available, the rerun summary should record the measured result against both `0.420` and `0.849`.
