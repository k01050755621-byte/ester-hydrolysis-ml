# -*- coding: utf-8 -*-
"""DFT 校准扩容：候选挑选（分层抽样，不看模型预测，只看 xTB 标签与化学类别）。

目标：把 xTB↔DFT 校准集从 6 个扩到 18 个左右，使 MAE 有统计意义，
并能分别报"强负区 / 近零区 / 正区"与"酚酯 / 醇酯""无卤 / 含卤"的误差。

约束：parent 原子数 ≤ 70（控机时）；同一 building-block 组合不重复；
      已校准的 6 个排除；强制纳入排序测定组里已标注的两支（C12225 / C8188）。
输出：fix_20260928/calib_expand/selection.csv + 屏幕表格。
"""
import os, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
ML = 'fix_20260928/ml_final'
OUT = 'fix_20260928/calib_expand'; os.makedirs(OUT, exist_ok=True)

DONE = ['C1074', 'C6533', 'C1223', 'C9132', 'C4141', 'C2606']
FORCE = ['C12225', 'C8188']          # 排序测定组成员（已标注）
MAX_ATOMS = 70
N_TARGET = 12                        # 新增数量（含 FORCE）

lab = pd.read_csv(f'{ML}/labels_final.csv')
man = json.load(open(f'{ML}/feature_manifest.json'))
MECH = man['mech_cols'] + ['分子量']
fv = pd.read_csv(f'{ML}/feat_vlib.csv', usecols=['编号'] + MECH)
smi = pd.read_csv(f'{ML}/vlib_predictions_v6.csv', usecols=['编号', '名称', 'SMILES'])
d = lab.merge(fv, on='编号', how='left').merge(smi, on='编号', how='left')
def natoms(s):
    if not isinstance(s, str):
        return 999
    m = Chem.MolFromSmiles(s)
    return Chem.AddHs(m).GetNumAtoms() if m else 999

d['n_atoms'] = d['SMILES'].map(natoms)
d['酚酯'] = d['n_alkAr'] > 0
d['含卤'] = d['n_halogen'] > 0
d = d[d['n_atoms'] <= MAX_ATOMS]
pool = d[~d['编号'].isin(DONE)].copy()
print(f'候选池（≤{MAX_ATOMS} 原子、未校准）= {len(pool)}')

# ---- 强制纳入 ----
forced = pool[pool['编号'].isin(FORCE)]
print('\n强制纳入（排序测定组）：')
print(forced[['编号', 'dE水解_kJmol', 'n_atoms', '酚酯', '含卤', '分子量']].to_string(index=False))

# ---- 分层抽样：xTB 标签 × (酚酯/醇酯) × (无卤/含卤) ----
bins = [(-100, -25, '强负'), (-25, -15, '负'), (-15, -5, '中偏负'),
        (-5, 5, '近零'), (5, 15, '正'), (15, 30, '强正'), (30, 100, '极正')]
rows = []
used_names = set()
for lo, hi, tag in bins:
    sub = pool[(pool['dE水解_kJmol'] > lo) & (pool['dE水解_kJmol'] <= hi)]
    for ph in [True, False]:                      # 酚酯 / 醇酯
        for hl in [False, True]:                  # 无卤优先
            s = sub[(sub['酚酯'] == ph) & (sub['含卤'] == hl)]
            s = s[~s['编号'].isin(FORCE)]
            if s.empty: continue
            # 优先：小分子（便宜）、且名称组合未用过
            s = s.sort_values('n_atoms')
            for _, r in s.iterrows():
                key = str(r['名称'])
                if key in used_names: continue
                used_names.add(key); rows.append({**r.to_dict(), '档': tag}); break

sel = pd.DataFrame(rows + [{'档': '强制', **f.to_dict()} for _, f in forced.iterrows()])
sel = sel.sort_values('dE水解_kJmol')
cols = ['编号', '名称', '档', 'dE水解_kJmol', 'n_atoms', '酚酯', '含卤', 'n_halogen', '分子量', 'SMILES']
sel[cols].to_csv(f'{OUT}/selection.csv', index=False)
print('\n==== 分层抽样结果 ====')
print(sel[cols[:-1]].to_string(index=False))
print(f'\n共 {len(sel)} 个（目标 {N_TARGET}）；原子数合计 {int(sel["n_atoms"].sum())}')
print(f'saved -> {OUT}/selection.csv')
