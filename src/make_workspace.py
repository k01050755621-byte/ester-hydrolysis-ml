#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble a runnable working tree from this repository.

The pipeline was developed with a fixed working-tree layout: all paths inside
the scripts are relative to a project root that contains the data files flat
plus the subfolders ``fix_20260928/ml_final``, ``fix_20260928/relabel_v7`` and
``fix_20260928/calib_expand`` (and a few scripts carry the developer's absolute
home path). This script recreates that layout inside a workspace directory of
your choice and rewrites the absolute paths accordingly — **the repository
copies stay untouched**.

Usage:
    python src/make_workspace.py /path/to/workspace

Then run stages exactly as documented in docs/REPRODUCING.md, e.g.:

    cd /path/to/workspace
    python fix_20260928/ml_final/ml11_retrain_v7.py     # ~40 min, CPU
"""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# absolute-path constants written during development -> rewritten on copy
OLD_PREFIX = "/Users/Zhuanz/polywork"
OLD_EXPAND = "os.path.expanduser('~/polywork')"

FLAT_ROOT = [  # -> workspace root
    "data/原料库_v1.xlsx",
    "data/dataset_v1.csv",
    "data/libraries/vlib_final.csv",
    "data/libraries/枚举库_ester_v1.csv",
    "data/libraries/枚举库_ester_v2.csv",
    "data/libraries/枚举库_ester_v3.csv",
    "data/libraries/枚举库_ester_v4.csv",
    "data/labels/ester_hyd.csv",
    "data/labels/al_round3_labels.csv",
    "data/dft/calib_ester_results.csv",
]

ENUM = [  # -> workspace root (they resolve everything against `base`)
    "src/02_enumeration/enum_ester.py",
    "src/02_enumeration/enum_ester2.py",
    "src/02_enumeration/enum_ester3.py",
    "src/02_enumeration/enum_ester4.py",
]

SRC_TO_MLFINAL = [  # -> fix_20260928/ml_final
    "src/03_features/build_features.py",
    "src/04_active_learning/al_round_generic.py",
    "src/04_active_learning/al4_query.py",
    "src/05_model_lineage/ml2_ablation.py",
    "src/05_model_lineage/ml3_designrules.py",
    "src/05_model_lineage/ml4_final_model.py",
    "src/05_model_lineage/ml5_learn_select.py",
    "src/05_model_lineage/ml7_retrain_v5.py",
    "src/05_model_lineage/ml8_retrain_v6.py",
    "src/05_model_lineage/ml9_robustness_final.py",
    "src/05_model_lineage/ml10_isomer_final.py",
    "src/06_finalize_v7/ml11_retrain_v7.py",
    "src/06_finalize_v7/ml12_v7_finalize.py",
    "src/06_finalize_v7/ml13_v7_extras.py",
    "src/07_expansion_100k/ml17_expand100k.py",
    "src/07_expansion_100k/check_100k.py",
    "src/08_candidates/exp1_candidates_v7.py",
    "src/10_figures/ml6_figures.py",
    "src/10_figures/ml14_fig2_v7.py",
    "src/10_figures/ml15_fig7_9_v7.py",
    "src/10_figures/ml16_fig1_v7.py",
]

# curated final outputs + lineage artifacts that later stages read as inputs
MLFINAL_DATA = [
    "data/lineage/ml_final",  # everything small from the original folder
    "data/predictions/vlib_predictions_v7.csv",
    "data/predictions/vlib_100k_predictions_v7.csv",
    "data/labels/labels_final_v7.csv",
    "data/labels/labels_v7_final.csv",
    "data/labels/labels_452.csv",
    "data/labels/label_ladder_v7.csv",
    "data/labels/al4_query.csv",
    "data/labels/al4_labels.csv",
    "data/labels/al5_query.csv",
    "data/labels/al5_labels.csv",
    "data/summaries/expand100k_summary.json",
]

RELABEL = [
    "src/01_labels/relabel_v7.py",
    "data/lineage/relabel_v7/labels_v7.csv",
]

CALIB = [
    "src/01_labels/al_round_label_generic.py",
    "src/01_labels/al4_label.py",
    "src/09_dft_calibration/run_calib_expand.py",
    "src/09_dft_calibration/select_candidates.py",
    "src/09_dft_calibration/validate_internal_opt.py",
    "src/09_dft_calibration/probe_conformer_noise.py",
    "src/09_dft_calibration/probe_hydrolysis_site.py",
    "src/09_dft_calibration/start_dft.sh",
    "src/09_dft_calibration/supervise_dft.sh",
    "data/lineage/calib_expand",
]

FIX_ROOT = [  # -> fix_20260928/ (candidate tables read by check_100k / exp1)
    "data/candidates/exp1_candidate_tiers_v7.csv",
]


def smart_copy(src: Path, dst: Path, ws: str) -> None:
    """Copy one file; if text, rewrite developer paths to the workspace."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(src, dst)
        return
    if OLD_PREFIX in text:
        text = text.replace(OLD_PREFIX + "/", ws + "/").replace(OLD_PREFIX, ws.rstrip("/"))
    if OLD_EXPAND in text:
        text = text.replace(OLD_EXPAND, repr(ws))
    dst.write_text(text, encoding="utf-8")
    if src.suffix == ".sh":
        dst.chmod(0o755)


def smart_copytree(src: Path, dst: Path, ws: str) -> None:
    for f in sorted(src.iterdir()):
        if f.is_dir():
            smart_copytree(f, dst / f.name, ws)
        else:
            smart_copy(f, dst / f.name, ws)


def main(ws_str: str) -> None:
    ws_path = Path(ws_str).resolve()
    if ws_path.exists() and any(ws_path.iterdir()):
        sys.exit(f"refusing: {ws_path} exists and is not empty")
    ws = str(ws_path)
    n = 0

    def place(rel: str, target_dir: Path) -> None:
        nonlocal n
        src = REPO / rel
        smart_copy(src, target_dir / src.name, ws)
        n += 1

    for rel in FLAT_ROOT + ENUM:
        place(rel, ws_path)
    for rel in SRC_TO_MLFINAL:
        place(rel, ws_path / "fix_20260928" / "ml_final")
    for rel in MLFINAL_DATA:
        src = REPO / rel
        if src.is_dir():
            smart_copytree(src, ws_path / "fix_20260928" / "ml_final", ws)
            n += 1
        else:
            place(rel, ws_path / "fix_20260928" / "ml_final")
    for rel in RELABEL:
        place(rel, ws_path / "fix_20260928" / "relabel_v7")
    for rel in CALIB:
        src = REPO / rel
        if src.is_dir():
            smart_copytree(src, ws_path / "fix_20260928" / "calib_expand", ws)
            n += 1
        else:
            place(rel, ws_path / "fix_20260928" / "calib_expand")
    for rel in FIX_ROOT:
        place(rel, ws_path / "fix_20260928")
    # ranking set: stored as .xlsx (true format), scripts expect the legacy .csv name
    smart_copy(REPO / "data/candidates/exp1_ranking_set_v7.xlsx",
               ws_path / "fix_20260928" / "exp1_ranking_set_v7.csv", ws)
    n += 1

    print(f"workspace ready: {ws}  ({n} items copied, paths rewritten)")
    print("next: cd " + ws + " && python fix_20260928/ml_final/check_100k.py")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
