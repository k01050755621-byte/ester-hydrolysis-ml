# ester-hydrolysis-ml

**Machine-learning screening of hydrolysable diester monomers with conformer-denoised semiempirical labels, mechanistic descriptors, and conformal prediction — over a 100,000-molecule virtual library built from commercially available building blocks.**

This repository accompanies a manuscript in preparation (preprint expected January 2027).
It releases the **complete, deterministic pipeline**: virtual library enumeration, GFN2-xTB label generation, feature engineering, ensemble learning with strict group validation, active learning, conformal prediction, DFT calibration, and candidate selection.

## Why this pipeline is different

1. **Conformer-denoised labels.** Single-conformer GFN2-xTB hydrolysis energies carry σ ≈ 12 kJ/mol of embedding noise (ETKDGv3 with default random seed). We quantify this noise, show it sets the floor on model accuracy, and fix it with a seeded multi-conformer protocol (6 conformers per species → GFN2 optimization → lowest energy). Under the strictest validation grouping, this single change lowers MAE by 19%.
2. **Mechanistic descriptors cure position-isomer blindness.** Whole-molecule descriptors assign identical values to positional isomers (2,367 distinct predictions for 12,380 molecules). Ester-bond microenvironment descriptors restore resolution (11,983 distinct predictions; within-family ranking ρ = 0.79, median 0.90).
3. **Honest uncertainty.** Split conformal prediction with category-conditional (Mondrian) intervals; every screening decision is reported with its reliability. 1,370 of 100,000 molecules have an entirely negative 90% prediction interval.
4. **Quantitative design rules.** Phenolic esters hydrolyze 15–18 kJ/mol more readily than aliphatic alcohol esters; each additional aryl halogen contributes ≈ −4.3 kJ/mol. Both rules survive domain shift into the expanded library.

## Key results (final v7 protocol)

| Metric | Value |
|---|---|
| Labeled compounds (GFN2-xTB, multi-conformer) | 1,099 |
| Virtual library (enumerated from commercial building blocks) | 100,000 |
| Scaffold-grouped CV (XGB+RF+HistGBDT ensemble, 5-fold × 3 seeds) | MAE 6.18 / R² 0.677 / Spearman ρ 0.795 |
| Strictest Butina grouping (Tanimoto 0.45, 130 clusters) | MAE 5.18 |
| Prospective active-learning rounds (rank correlation) | 0.797 → 0.824 → 0.847 |
| Conformal (global q90 / empirical coverage) | 14.53 kJ/mol / 91.0% |
| DFT calibration (19 molecules, wB97X-D3/def2-TZVP//def2-SVP) | r = 0.971; corrected residual MAE 3.96 kJ/mol |
| Screening output (90% CI entirely negative, 100k library) | 1,370 molecules |

## Pipeline

```
commercial building blocks (原料库_v1.xlsx)
        │  02_enumeration  (9 substituents × single/double decoration, seeded sampling)
        ▼
virtual library 100,000 diesters ──────────────► data/libraries/
        │  01_labels  (GFN2-xTB, 6 seeded conformers/species, lowest energy)
        ▼
hydrolysis-energy labels 1,099 ────────────────► data/labels/
        │  03_features (base + mechanistic + Morgan + MACCS)
        ▼
feature matrix ──► 04_active_learning (3 rounds, prospective)
        │           05_model_lineage (ablation, design rules, robustness,
        │                          isomer resolution, v4→v7 label A/B)
        ▼
final ensemble (06_finalize_v7) ──► predictions + conformal + Mondrian
        │  07_expansion_100k (feature rebuild + rescreen + consistency checks)
        │  08_candidates (tiers + 5-member experimental ranking set)
        ▼
09_dft_calibration (19 molecules, psi4) ──► linear correction, r = 0.971
```

## Repository layout

```
data/
  原料库_v1.xlsx          building-block source library (commercially purchasable)
  dataset_v1.csv          489 collected compounds (BDE / hydrolysis literature labels)
  labels/                 452 seed labels → v7 final 1,099 labels + AL round queries
  libraries/              enumerated virtual libraries v1–v4 (v4 = 87,620 molecules)
  predictions/            v7 predictions, 12,380 + 100,000 molecule versions
  summaries/              all result summaries (JSON/CSV) cited in the manuscript
  candidates/             screening tiers, Pareto set, halogen-free top-10, ranking set
  dft/                    19-molecule DFT calibration table + conformer-noise probe
  lineage/                intermediate-version artifacts (v4–v6 predictions, summaries,
                          AL round tables) kept so every later stage can re-run
src/
  make_workspace.py       one command to assemble a runnable working tree
  01_labels … 10_figures  pipeline scripts, numbered in execution order
docs/
  REPRODUCING.md          how to re-run everything (order, seeds, runtimes, binaries)
  DATA_DICTIONARY.md      column-level description of every released data file
```

## Quickstart

```bash
# environment (Python 3.11)
pip install -r requirements.txt

# external binaries (labels / DFT calibration only)
#   xtb  >= 6.7   (GFN2-xTB)      conda install -c conda-forge xtb
#   psi4  = 1.11  (DFT)           conda install -c conda-forge psi4

# assemble a runnable workspace (rewrites paths automatically; repo stays untouched)
python src/make_workspace.py /path/to/workspace
cd /path/to/workspace

# audit the released 100k output — 28 invariants, all must PASS (~5 min)
python fix_20260928/ml_final/check_100k.py

# rebuild features for the 12,380-molecule library (CPU, ~20 s, bit-identical)
python fix_20260928/ml_final/build_features.py

# retrain the final v7 model + full evaluation (CPU, ~40 min)
python fix_20260928/ml_final/ml11_retrain_v7.py

# rescreen the 100k library (CPU, ~10 min)
python fix_20260928/ml_final/ml17_expand100k.py
```

All sampling is seeded (`RandomState(42)`, fixed ETKDGv3 seeds) and bit-for-bit reproducible; the 100k enumeration was verified by full re-run with identical MD5. See `docs/REPRODUCING.md` for the complete stage-by-stage guide.

## Data availability

All labels, predictions, and screening outputs are in `data/` (total ~37 MB). Feature matrices are **not** shipped (230 MB) but are regenerated deterministically by `src/03_features/build_features.py` and `src/07_expansion_100k/ml17_expand100k.py`.

## Citation

A preprint will be available on ChemRxiv/arXiv in January 2027. Until then, please cite this repository:

> Ziyu Xu (Shandong University), *ester-hydrolysis-ml*, 2026. https://github.com/<account>/ester-hydrolysis-ml

## License

Code and data are released under [CC BY-NC 4.0](LICENSE) (Attribution-NonCommercial). Commercial use requires separate permission from the author.
