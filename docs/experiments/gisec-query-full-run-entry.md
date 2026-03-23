# GISEC Query Alpha Full-Run Entry Rules

full runs are forbidden until the previous phase passes its relative gate.

## Entry Conditions

- `UQ` must first prove that the object-first base is stronger than `v1.5 legacy`.
- `UR` may only enter full runs after the query-only base is stable.
- `UG` may only enter full runs after the query-only base is stable.
- the combined system must still contain both `reference` and `graph` before final paper claims are made

## Non-Conditions

- A full run is not allowed only because GPU is available.
- A large run is not allowed only because a single short-run number looks pretty.

## Promotion Logic

- each previous phase passes its relative gate
- then the next phase may request a larger run
- otherwise the project stays in short-run diagnosis and refinement mode
