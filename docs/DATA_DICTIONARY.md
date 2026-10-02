# Data dictionary

Column-level description of every released data file. Molecule IDs are `C####`
(integers after "C"); energies in kJ/mol unless noted.

## data/ (root)

| File | Content |
|---|---|
| `原料库_v1.xlsx` | Building-block source library (commercially purchasable diester compounds). Columns: `编号` (ID), `SMILES`, plus metadata. The enumeration pipeline cuts these into cores (two ester-adjacent attach points) and caps (one attach point). |
| `dataset_v1.csv` | 489 collected compounds with literature labels: 164 BDE labels + 185 hydrolysis-energy labels, plus 48 epoxy monomers/curing agents and dynamic-bond model compounds (ester / disulfide / imine / acetal / ketal / siloxane). Used as the seed source for `labels_452.csv`. |

## data/labels/

| File | Content |
|---|---|
| `labels_452.csv` | 452 seed hydrolysis labels (编号 / SMILES / 来源 source / label) from stage-02 feature factory. |
| `labels_final_v7.csv` | **Final 1,099 labels (v7 protocol)**: `编号, dE水解_kJmol, 来源` |
| `labels_v7_final.csv` | Same 1,099 with legacy comparison: `编号, dE_v7, dE_old, 差_新旧, 来源` — the A/B pair behind the conformer-noise finding |
| `label_ladder_v7.csv` | Label counts per round (452 → AL3 → AL4 → AL5 = 1,099) |
| `al4_query.csv`, `al4_labels.csv`, `al5_query.csv`, `al5_labels.csv` | Active-learning round 4/5 query sets and their subsequently measured labels |
| `ester_hyd.csv` | Literature hydrolysis-energy label source (seed stage input) |
| `al_round3_labels.csv` | Active-learning round-3 labels (seed stage input) |

## data/lineage/

| File | Content |
|---|---|
| `ml_final/` | Everything small from the original working folder: v4/v5/v6 predictions and summaries, final-stage aggregates (`final_models.csv`, `final_ablation.csv`, `final_butina.csv`, `final_yrandom.json`), isomer and design-rule summaries, `rule_crosstab.csv`, `labels_772.csv`, and the v6 label set (`labels_final.csv`). These are the inputs consumed by `ml11/ml12/ml13/ml17` — kept so every later stage can re-run. |
| `relabel_v7/labels_v7.csv` | Raw output of the v7 relabeling protocol (with per-species conformer spans), input to `ml11`. |
| `calib_expand/` | DFT calibration working tables: `selection.csv` (19 chosen molecules), `results_w*.csv` (per-worker psi4 results), `site_ambiguity.csv`. |

## data/libraries/

| File | Content |
|---|---|
| `枚举库_ester_v1.csv` … `枚举库_ester_v4.csv` | Enumerated virtual libraries. Columns: `编号, 角色, 名称` (decoded decoration recipe, e.g. `D(C328_F4xPh_Cl3xPh_Br3_F6)` = core C328 with F at position 4×Ph ring, etc.), `SMILES`, `动态键类型` (always 酯键/ester), `文献来源DOI`, `备注`. v1–v3 = 12,380 molecules total (seeded sampling from the 6-substituent space); **v4 = 87,620 new molecules** (9 substituents, single+double decoration, alkyl caps C1–C10; IDs C12884–C100503). |
| `vlib_final.csv` | The 12,380-molecule library with full metadata as consumed by the modeling stages (`编号, 角色, 名称, SMILES, 分子量, …`). |

## data/predictions/

| File | Content |
|---|---|
| `vlib_predictions_v7.csv` | Final-model predictions on the 12,380-molecule library: `编号, 名称, SMILES, 分子量` (MW), `预测_v7` (point prediction), `q90_v7`, `CI下_v7` / `CI上_v7` (90% conformal bounds), `n_hal` (halogen count) |
| `vlib_100k_predictions_v7.csv` | Same columns + `库` (= v1-3 for the original 12,380 / v4 for the 87,620 new) on the full 100,000-molecule library |

## data/summaries/

| File | Content |
|---|---|
| `v7_final_summary.json` | Master summary: CV metrics, prospective rounds, conformal q90/coverage, ablation, feature sets, Butina clusters, design rules |
| `expand100k_summary.json` | 100k-expansion statistics (CI-negative counts, halogen-free counts, new-molecule extremes, top-15 new molecules) |
| `feature_manifest.json` | Feature groups (base / mechanistic / Morgan / MACCS) with exact column names — load this in every modeling script |
| `mondrian_v7.csv` / `.json` | Category-conditional (Al-Al / Al-Ar / Ar-Al / Ar-Ar) conformal q90 and coverage |
| `butina_clusters_v7.json` | Butina clustering at Tanimoto 0.45: 130 clusters, max 73, mean 8.45 |
| `learning_curve_v7.csv` | Learning curve (label count vs. strictest-group MAE) |
| `yrandom_table.json` | y-randomization control results |

## data/candidates/

| File | Content |
|---|---|
| `exp1_candidate_tiers_v7.csv` | Tiered experimental candidates: strong 143 / medium 140 / weak 2,327 |
| `exp1_ranking_set_v7.xlsx` | **5-member experimental ranking set** spanning the prediction spectrum (C2001 / C12225 / C8986 / C8313 / C10640). Note: Excel format. |
| `pareto_v7.csv` | Prediction-vs-MW Pareto front (4 members) |
| `all_negative_CI_v7.csv` | Molecules whose entire 90% CI is negative (12,380 library: 143) |
| `halogen_free_top10_v7.csv` | Top-10 halogen-free candidates |

## data/dft/

| File | Content |
|---|---|
| `calib_19_merged.csv` | 19-molecule DFT calibration: GFN2 vs. wB97X-D3/def2-TZVP//def2-SVP gas-phase hydrolysis energies, per-molecule deviations |
| `conformer_noise.csv` | Single-conformer embedding-noise measurements (same molecule, repeated embeds → ΔE spread; σ ≈ 12 kJ/mol) |
