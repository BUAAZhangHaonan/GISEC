# GISEC Query Alpha Experiment Ladder

`GISEC Query Alpha` uses one fixed experiment order.

## Official Order

1. `v1.5 legacy` historical baseline
2. `UQ-s`
3. `UQ-m`
4. `UR-*`
5. `UG-*`
6. `UA-*`

## Meaning

- `UQ-s`
  - query-only object-first baseline, small scale
- `UQ-m`
  - query-only object-first baseline, medium scale
- `UR-*`
  - query-only base plus rescue-side reference
- `UG-*`
  - query-only base plus local graph rescue
- `UA-*`
  - query-only base plus both rescue modules

## Ordering Rule

- `UQ-s` and `UQ-m` must be evaluated first.
- `UR-*` cannot be opened until the query-only base is stable.
- `UG-*` cannot be opened until the query-only base is stable.
- `UA-*` only opens after both `reference` and `graph` have individually earned promotion.

This ladder is intentional. It prevents the project from hiding a weak object-first base behind rescue modules too early.
