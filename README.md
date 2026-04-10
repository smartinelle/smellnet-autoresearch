# smellnet-autoresearch

Standalone autoresearch and benchmark tooling for SmellNet-style sensor classification.

This scaffold is intended to hold the original experimentation workflow separately from the inherited `SmellNet` fork. It focuses on:

- exact-upstream benchmark-faithful preprocessing
- supervised transformer baselines
- grouped-validation search loops
- an initial contrastive GC-MS search loop
- explicit artifact contracts for edge deployment

## Scope

This repo is meant to contain the tooling and documentation around the experiments, not dataset snapshots or large run directories.

Expected external data layout:

- `data/offline_training`
- `data/offline_testing`
- `data/gcms_dataframe.csv`

## Package

The main package is `smellnet_autoresearch`.

Install locally with:

```bash
python3 -m pip install -e ".[dev]"
```

Primary entrypoints:

- `python -m smellnet_autoresearch.train`
- `python -m smellnet_autoresearch.search`
- `python -m smellnet_autoresearch.search_phase2`
- `python -m smellnet_autoresearch.search_contrastive`

## Notes

- `docs/findings.md` captures the benchmark and autoresearch results gathered so far.
- `docs/publishing_notes.md` captures the legal and publication split strategy.
- `openspec/` carries the current design/spec notes for the harness.
- The strongest completed supervised run so far is the phase-2 exact-upstream search.
- The contrastive path is included as an initial extension track, but it has only been lightly explored so far.
