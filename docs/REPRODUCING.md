# Conformer-denoised labels — complete reproduction guide

This guide re-runs the entire pipeline end to end. Everything is deterministic
(seeded); two independent full re-runs of the 100k enumeration produced
byte-identical output (MD5-verified).

## Environment

| Component | Version used | Notes |
|---|---|---|
| Python | 3.11 | `pip install -r requirements.txt` |
| RDKit | 2026.03.5 | features, enumeration, SMARTS |
| XGBoost / scikit-learn | 3.2.0 / 1.9.0 | models |
| xtb | 6.7.1 | GFN2 labels (stage 01) |
| psi4 | 1.11 | DFT calibration only (stage 09), `wB97X-D3/def2-SVP` opt + `def2-TZVP` single point, gas phase |

Hardware: everything ran on a 10-core laptop CPU. No GPU required.

## Path setup (one-time)

The pipeline was developed in a fixed working tree; scripts resolve data
relative to a project root that also holds the subfolders
`fix_20260928/{ml_final,relabel_v7,calib_expand}`. **Do not run the scripts
inside the repository** — instead, assemble a runnable workspace (paths are
rewritten automatically, repository copies stay untouched):

```bash
python src/make_workspace.py /path/to/workspace
cd /path/to/workspace
```

All commands below assume `cwd = workspace`. This was tested end-to-end:
`check_100k.py` passes all invariants and `build_features.py` regenerates the
feature matrix **bit-identically** (12,380 × 1,238, max diff 0.0).

## Stage order and runtimes

| # | Stage | Script(s) (inside workspace) | Runtime | Output |
|---|---|---|---|---|
| 01 | Enumeration v1–v4 | `enum_ester{,2,3,4}.py` | 4 min (v4) | `枚举库_ester_v4.csv` at workspace root (regenerates v4 bit-identically) |
| 02 | Feature factory | `fix_20260928/ml_final/build_features.py` | ~20 s | `feat_vlib.csv`, `feature_manifest.json`, `labels_452.csv` |
| 03 | Ablation + design rules + robustness + isomer resolution | `fix_20260928/ml_final/ml{2,3,4,5,9,10}_*.py` | 10–60 min each | per-stage JSON/CSV in the same folder |
| 04 | Label lineage v4→v6 (single-conformer) | `ml{7,8}_*.py` | ~40 min each | versioned predictions (kept for the A/B narrative) |
| 05 | **v7 relabeling** (multi-conformer GFN2) | `fix_20260928/relabel_v7/relabel_v7.py` | ~30 h × 1,099 labels (parallelizable per molecule; interruptible with per-conformer cache) | `labels_v7.csv` + best-conformer geometries |
| 06 | Final v7 model + A/B | `ml11_retrain_v7.py` | ~40 min | `vlib_predictions_v7.csv`, `v7_summary.json` |
| 07 | Finalize (screening outputs, Pareto, CI-negative) | `ml12_v7_finalize.py` | ~5 min | summaries + candidate tables |
| 08 | Extras (Mondrian, learning curve, Butina clusters) | `ml13_v7_extras.py` | ~20 min | `mondrian_v7.*`, `learning_curve_v7.csv`, `butina_clusters_v7.json` |
| 09 | 100k expansion + rescreening | `ml17_expand100k.py` | ~10 min | `vlib_100k_predictions_v7.csv`, `expand100k_summary.json` |
| 10 | Consistency audit (read-only) | `check_100k.py` | ~5 min | PASS/FAIL report (all PASS at release) |
| 11 | Experimental candidates | `exp1_candidates_v7.py` | <1 min | tiers + 5-member ranking set |
| 12 | DFT calibration (19 molecules) | `fix_20260928/calib_expand/{select_candidates,run_calib_expand}.py` + `start_dft.sh` | ~100 h total CPU (parallelizable, ~5 h/molecule) | `calib_19_merged.csv` |
| 13 | Manuscript figures | `ml{6,14,15,16}_*.py` | ~5 min | figures |

## The v7 labeling protocol (what makes labels reproducible)

1. For each species (parent / hydrolysis product P1 / P2): embed **6
   conformers** with `AllChem.EmbedMultipleConfs(..., ETKDGv3(), randomSeed=<fixed>)`.
2. Optimize each conformer with `xtb --opt --gfn 2`; keep the **lowest-energy**
   conformer per species.
3. ΔE(hydrolysis) = E(P1)min + E(P2)min − E(parent)min − E(H2O,min).
4. Water is labeled once with the same protocol and shared globally.
5. Hydrolysis site = first SMARTS match (identical to legacy labels, keeps A/B
   comparable).

The legacy single-conformer protocol (kept in the lineage scripts as the v6
arm) embeds one conformer with `randomSeed=-1`, which alone injects
σ ≈ 12 kJ/mol of noise — see `data/dft/conformer_noise.csv` and the A/B table
in `data/summaries/v7_final_summary.json`.

## Known pitfalls (found the hard way, all fixed in the released scripts)

- **Silent row drops**: any merge on molecule IDs must assert
  `len(output) == len(input)`; SMILES/meta are always multi-source merged
  before filtering (100 small esters exist only in `dataset_v1.csv`).
- **PCM total energies**: implicit-solvent single points with
  `pcm_scf_type='total'` return interaction energies, not total energies. The
  calibration therefore uses **gas phase** DFT with GFN2 gas-phase references.
- **Figure scripts reading stale summaries**: every figure script pins its
  data source explicitly; after any protocol change, grep the script for its
  input file before trusting the output.

## Verifying your re-run

`src/07_expansion_100k/check_100k.py` re-validates 28 invariants of the
released 100k output (row counts, ID continuity, SMILES uniqueness, RI-Valid
spot checks, old-vs-new prediction agreement to 1e-8, ranking-set stability).
It must print `全部检查通过` / all-PASS on the released data; if you regenerate
the data, run it again.
