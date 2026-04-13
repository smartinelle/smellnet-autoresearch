# smellnet-autoresearch

**Training + autoresearch harness for SmellNet-style sensor classifiers — benchmark-faithful reproduction, validation-locked hyperparameter search, and edge-export for Raspberry Pi inference.**

This repo is the **training-side sibling** of [`smell-pi`](https://github.com/smartinelle/smell-pi). `smell-pi` is the hardware port of [SmellNet (Feng et al., 2025)](https://arxiv.org/abs/2506.00239) — the physical sensor rig and pure-Python collection stack. `smellnet-autoresearch` is where the models get trained, searched, and exported as edge-ready checkpoint bundles that `smell-pi` ships in `artifacts/`. Both repos stand alone, but together they cover the full pipeline: physical rig → raw sensor CSVs → benchmark-faithful training → exported checkpoint → on-device inference.

The current headline result is a supervised 6-channel Transformer reaching **57.97% Top-1 / 88.05% Top-5** on the SmellNet offline test set via a two-stage validation-locked search, matching the paper's logged 58.5% Top-1 best.

- **Paper**: [SMELLNET: A Large-scale Dataset for Real-world Smell Recognition](https://arxiv.org/abs/2506.00239) (Feng et al., 2025)
- **Upstream repo**: https://github.com/MIT-MI/SmellNet
- **Dataset**: [SmellNet on Hugging Face](https://huggingface.co/datasets/DeweiFeng/smell-net)
- **Sibling repo** (hardware + edge inference): [`smell-pi`](https://github.com/smartinelle/smell-pi)

Not affiliated with or endorsed by the SmellNet authors. The dataset, the original sensor collection methodology, and the ScentFormer architecture all belong to the SmellNet team.

---

## What's in the repo today

`smellnet-autoresearch` is a standalone Python package (`smellnet_autoresearch/`) with three training tracks:

1. **Exact-upstream supervised baseline** — `train.py`. A benchmark-faithful reproduction of the paper's 6-channel sensor-only single-model Transformer: same preprocessing (baseline-subtract → FOTD with lag 25 → sliding window 100 / stride 50), same 6-channel subset (`NO2, C2H5OH, VOC, CO, Alcohol, LPG`), same official `offline_training` / `offline_testing` split.
2. **Validation-locked autoresearch search** — `search.py` (phase 1) and `search_phase2.py` (phase 2). Proper experimental hygiene: grouped file-level validation folds carved out of training, search objective evaluated on validation only, test set touched once at the very end. Phase 2 adds a two-stage pipeline with promotion from cheap stage-1 trials to longer stage-2 runs, plus extra knobs (weight decay, label smoothing, warmup ratio).
3. **Contrastive GC-MS extension** — `search_contrastive.py`. An initial extension track that brings the paper's chemical-composition modality into the loss. Under the short search budget used so far it underperforms the supervised track; it's shipped as an extension point, not a benchmark anchor.

Full run results and interpretation live in [`docs/findings.md`](docs/findings.md).

---

## Headline results

All runs use the exact-upstream 6-channel sensor-only contract and grouped file-level validation. Test metrics touch `offline_testing` once at the end.

| Track | Trials | Best config | Test Top-1 | Test Top-5 |
|---|---|---|---|---|
| **Phase 2 supervised** (current anchor) | 14 stage-1 → 1 promoted | `model_dim=512, heads=8, layers=6, dropout=0.05, lr=3e-4` | **57.97** | **88.05** |
| Phase 1 supervised | 127 | `model_dim=512, heads=8, layers=5, dropout=0.0, lr=1e-4` | 56.57 | 87.65 |
| Exact-upstream smoke baseline | 1 (5 epochs) | default | 52.99 | 89.24 |
| Contrastive (extension track) | 30 | `model_dim=256, heads=8, layers=2, dropout=0.05, lr=3e-4, temp=0.1` | 50.40 | 85.26 |

**For context:** the SmellNet paper's best publicly logged sensor-only Transformer run hit **58.57 Top-1**. The phase-2 result here (57.97) reproduces that performance band from scratch, under validation-locked search, with a checkpoint that actually saves (the upstream repo logs a checkpoint path but the main runner never calls `torch.save` — see [`docs/findings.md §5`](docs/findings.md)).

---

## Methodological notes

A few deliberate experimental-hygiene choices that matter more than the raw numbers:

- **Grouped file-level validation.** Windows from the same recording are highly correlated. Splitting at the window level leaks. This repo carves validation at the file level, holding out whole CSVs per class, so search and checkpoint selection can't cheat by memorising neighbours of a training window.
- **Validation-locked search.** Every hyperparameter decision and every checkpoint selection happens on the validation folds. The held-out test set is touched exactly once, at the end of each search phase, to produce the numbers in the table above.
- **Window-level vs file-level are not the same task.** `docs/findings.md` goes into this in detail. The benchmark anchor here is window-level, single-model, no aggregation, no ensemble — directly comparable to the paper's published number. File-level and ensemble results, when added, live in a separate track and don't get mixed into the benchmark table.
- **Two-stage search with promotion.** Phase 2 runs cheap short trials first, then promotes the best configs into longer, multi-fold runs. This spends compute where it matters without exploding the search surface.
- **Honest about the contrastive gap.** The short contrastive search budget used here is not comparable to the paper's longer contrastive runs. The fact that it underperforms supervised is reported as-is, not talked around.

---

## Install & run

Clone and install as an editable package:

```bash
python3 -m pip install -e ".[dev]"
```

Expected external data layout (download from the Hugging Face dataset, then point the scripts at it):

```
data/
├── offline_training/
├── offline_testing/
└── gcms_dataframe.csv
```

Entry points:

```bash
# Exact-upstream baseline training (single model)
python -m smellnet_autoresearch.train \
    --train-dir data/offline_training \
    --test-dir  data/offline_testing \
    --output-dir runs/baseline

# Phase-1 validation-locked search
python -m smellnet_autoresearch.search

# Phase-2 two-stage search with promotion (current anchor)
python -m smellnet_autoresearch.search_phase2

# Contrastive GC-MS extension track
python -m smellnet_autoresearch.search_contrastive
```

See [`docs/findings.md`](docs/findings.md) for the settings each entry point was run with and the full results.

---

## Exported artifact contract

Every training run writes an edge-ready bundle into its output directory:

- `checkpoint.pt` — model weights
- `labels.json` — class label order
- `preprocessing.json` — `raw_sensor_columns`, `used_sensor_columns`, `dropped_sensor_columns`, `gradient_period`, `window_size`, `stride`, `scaler_mean`, `scaler_scale`, `model` config
- `training_metrics.json` / `final_test_metrics.json` — per-epoch and final test numbers

This bundle layout is what the sibling repo [`smell-pi`](https://github.com/smartinelle/smell-pi) consumes under `artifacts/`. The `preprocessing.json` is the authoritative input contract for running the checkpoint on new data — the ScentFormer architecture, the column subset, and the scaler parameters are all pinned there, so a downstream consumer doesn't have to reconstruct them from this repo's source.

---

## Repo layout

```
smellnet-autoresearch/
├── README.md
├── LICENSE                          # MIT
├── pyproject.toml
├── smellnet_autoresearch/           # main Python package
│   ├── datasets.py                  # CSV loader, windowing, grouped splits
│   ├── prepare.py                   # preprocessing contract (baseline sub + FOTD + windows + scaler)
│   ├── model_zoo.py                 # ScentFormer + baseline architectures
│   ├── losses.py                    # supervised + contrastive losses
│   ├── train.py                     # exact-upstream supervised baseline trainer
│   ├── search.py                    # phase 1 validation-locked search
│   ├── search_phase2.py             # phase 2 two-stage promoted search (current anchor)
│   ├── search_contrastive.py        # contrastive GC-MS extension track
│   ├── evaluation.py                # window-level + file-level evaluators
│   └── taxonomy.py                  # substance grouping + label utilities
├── tests/
├── docs/
│   ├── findings.md                  # detailed run results and interpretation
│   └── publishing_notes.md          # legal / publication split strategy
└── openspec/                        # in-progress design specs for new tracks
```

---

## Roadmap

- **Prefix evaluation.** Score the same trained models at first window / first 2 windows / first 3 windows / full file, so we can tell whether stronger file-level protocols still help in a streaming / early-prediction setting. Listed as "recommended next measurement" in [`docs/findings.md §9`](docs/findings.md).
- **Contrastive with a real budget.** The current contrastive search used a 1-hour budget and underperformed the supervised track. A proper long-run contrastive training pass is needed before the track can be fairly compared.
- **File-level and ensemble tracks.** Present as a separate evaluation path, explicitly not mixed into the window-level benchmark anchor.
- **Feeding `smell-pi`.** Tighter coupling between export format and the `smell-pi` inference path once the Pi-side calibration bridge is in place.

---

## Attribution & citation

All credit for the dataset, the sensor collection methodology, and the ScentFormer architecture belongs to the SmellNet authors. This repo is an independent training / autoresearch harness built around their public work.

```bibtex
@article{feng2025smellnet,
  title   = {SMELLNET: A Large-scale Dataset for Real-world Smell Recognition},
  author  = {Feng, Dewei and others},
  journal = {arXiv preprint arXiv:2506.00239},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.00239}
}
```

See the upstream [MIT-MI/SmellNet](https://github.com/MIT-MI/SmellNet) repo for dataset and baseline licensing. Code authored in `smellnet-autoresearch` is released under the MIT license in [`LICENSE`](LICENSE).
