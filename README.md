# smellnet-autoresearch

Standalone autoresearch and benchmark tooling for SmellNet-style sensor classification.

This repo holds an independent experimentation harness around the [SmellNet](https://github.com/MIT-MI/SmellNet) dataset — transformer baselines, grouped-validation search loops, a contrastive GC-MS track, and explicit artifact contracts for edge deployment. It was split out of a downstream fork of SmellNet so that the tooling can evolve separately from the upstream codebase and dataset snapshots.

## Upstream / Attribution

This project builds directly on **SmellNet** (Feng et al., 2025). All credit for the dataset, the original sensor collection pipeline, and the ScentFormer baseline goes to the SmellNet authors.

- **Original repo**: https://github.com/MIT-MI/SmellNet
- **Paper**: [SMELLNET: A Large-scale Dataset for Real-world Smell Recognition](https://arxiv.org/abs/2506.00239) (Feng et al., 2025)
- **Dataset**: [SmellNet on Hugging Face](https://huggingface.co/datasets/DeweiFeng/smell-net)

`smellnet-autoresearch` is **not** affiliated with or endorsed by the SmellNet authors. It is an independent research harness that reuses the dataset, benchmark protocol, and naming conventions from the original work.

## Scope

This repo contains tooling and documentation around the experiments — **not** dataset snapshots or large run directories. It focuses on:

- exact-upstream benchmark-faithful preprocessing
- supervised transformer baselines
- grouped-validation search loops
- an initial contrastive GC-MS search loop
- explicit artifact contracts for edge deployment

Expected external data layout (downloaded separately from Hugging Face):

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

## Citation

If you use this harness, please cite the original SmellNet paper:

```bibtex
@article{feng2025smellnet,
  title   = {SMELLNET: A Large-scale Dataset for Real-world Smell Recognition},
  author  = {Feng, Dewei and others},
  journal = {arXiv preprint arXiv:2506.00239},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.00239}
}
```
