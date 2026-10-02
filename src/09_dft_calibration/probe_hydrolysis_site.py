# -*- coding: utf-8 -*-
"""诊断：不对称二酯的"水解位点"歧义。

背景：calib_ester/*/xtb.log 重算出的 C6533 = -10.0、C1223 = -30.6，
但训练标签库（ester_hyd.csv / labels_final.csv）给的是 C6533 = -33.7、C1223 = +9.7。
parent 几何与库结构完全一致（已用 rdDetermineBonds 反推验证），
所以怀疑：两者切的是**不同的酯基**（不对称二酯有两个可切位点）。

做法：用同一 SMARTS 枚举**全部**水解位点的产物对，逐个做 xTB GFN2 优化，
按 xTB 口径算 ΔE = E(P1)+E(P2)-E(parent)-E(water)，看标签值落在哪个位点上。
输出：fix_20260928/calib_expand/site_ambiguity.csv
"""
import os, re, subprocess, itertools
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork')
OUT = os.path.join(HOME, 'fix_20260928', 'calib_expand', 'site_probe')
os.makedirs(OUT, exist_ok=True)
XTB = os.path.expanduser('~/miniconda3/envs/poly/bin/xtb')
KJ = 2625.5

RXN = ReactionFromSmarts('[#6:1](=[#8:2])-[#8;H0:3]-[#6:4]>>[#6:1](=[#8:2])-[OH].[#6:4]-[OH]')
v = pd.read_csv(os.path.join(HOME, 'vlib_final.csv')).set_index('编号')
lab = pd.read_csv(os.path.join(HOME, 'fix_20260928', 'ml_final', 'labels_final.csv')).set_index('编号')
cal = pd.read_csv(os.path.join(HOME, 'calib_ester_results.csv')).set_index('编号')


def xtb_e(name, smi_or_xyz, is_xyz=False):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    if not os.path.exists(os.path.join(d, 'xtb.log')):
        if is_xyz:
            open(os.path.join(d, 'start.xyz'), 'w').write(smi_or_xyz)
        else:
            m = Chem.AddHs(Chem.MolFromSmiles(smi_or_xyz))
            AllChem.EmbedMolecule(m, AllChem.ETKDGv3()); AllChem.MMFFOptimizeMolecule(m)
            Chem.MolToXYZFile(m, os.path.join(d, 'start.xyz'))
        with open(os.path.join(d, 'xtb.log'), 'w') as f:
            subprocess.run([XTB, 'start.xyz', '--opt', '--gfn', '2'], cwd=d,
                           stdout=f, stderr=subprocess.STDOUT)
    t = open(os.path.join(d, 'xtb.log')).read()
    return float(re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', t)[-1])


def xyz_block(p):
    L = open(p).read().splitlines()
    return '\n'.join(L[:1] + ['comment'] + L[2:2 + int(L[0])]) + '\n'


# 水（复用 calib_ester 的 xTB 水）
Xw = xtb_e('water', 'O')
print('water xTB =', Xw, flush=True)

IDS = ['C1074', 'C6533', 'C1223', 'C9132', 'C4141', 'C2606']
rows = []
for cid in IDS:
    smi = v.loc[cid, 'SMILES']
    mol = Chem.MolFromSmiles(smi)
    # parent：直接复用已有几何的 xTB 能量（一致口径）
    par_dir = os.path.join(HOME, 'calib_ester', f'{cid}_parent', 'xtb.log')
    if os.path.exists(par_dir):
        Ep = float(re.findall(r'TOTAL ENERGY\s+(-?\d+\.\d+)', open(par_dir).read())[-1])
    else:  # C2606 无 parent 目录时用 sp2_C2606 的几何
        gp = os.path.join(HOME, 'sp2_C2606', 'parent_geom.xyz')
        Ep = xtb_e(f'{cid}_parent_probe', xyz_block(gp) if os.path.exists(gp) else smi, is_xyz=os.path.exists(gp))
    sets = RXN.RunReactants((mol,))
    print(f'\n{cid}: parent xTB={Ep:.6f}  枚举到 {len(sets)} 个位点', flush=True)
    for k, ps in enumerate(sets):
        smis = []
        for p in ps:
            try:
                p.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(p, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except Exception:
                pass
            smis.append(Chem.MolToSmiles(p))
        E = [xtb_e(f'{cid}_site{k}_p{j}', s) for j, s in enumerate(smis)]
        dE = (sum(E) - Ep - Xw) * KJ
        rows.append({'编号': cid, '位点': k, '产物1': smis[0], '产物2': smis[1],
                     'xTB水解_kJmol': round(dE, 1),
                     '标签库': round(float(lab.loc[cid, 'dE水解_kJmol']), 1) if cid in lab.index else None,
                     '校准文件': round(float(cal.loc[cid, '水解_xTB_kJmol']), 1) if cid in cal.index else None})
        print(f'   位点{k}: {smis[0]} + {smis[1]}  ->  ΔE={dE:7.1f}', flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HOME, 'fix_20260928', 'calib_expand', 'site_ambiguity.csv'), index=False)
print('\n==== 汇总 ====')
print(d[['编号', '位点', 'xTB水解_kJmol', '标签库', '校准文件']].to_string(index=False))
