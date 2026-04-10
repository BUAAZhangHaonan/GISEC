# GISEC Release Checklist

## Repository Surface

- [ ] `README.md` reflects the final project name, environment, and commands
- [ ] `docs/new-session-handoff.md` points to the real repository and data roots
- [ ] all public scripts use `gisec` terminology consistently

## Reproducibility

- [ ] `environment.yml` installs successfully on a clean machine
- [ ] `conda run -n gisec pytest -q` passes
- [ ] the best-variant train/eval commands are documented and copy-pastable

## Results

- [ ] Stage 1 matrix summary exists under `docs/experiments/`
- [ ] extended metrics table exists under `docs/experiments/`
- [ ] diagnostics artifacts referenced in the paper exist under `output/analysis` or are summarized in `docs/results/`

## Publication Prep

- [ ] license chosen
- [ ] citation block added
- [ ] bridge status into `magformer` documented
- [ ] any private dataset assumptions are clearly marked
