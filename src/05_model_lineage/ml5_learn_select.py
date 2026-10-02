# -*- coding: utf-8 -*-
"""ML 补强 · 第 5 步：学习曲线 + 特征选择（紧凑可解释模型）

输出：
  learning_curve_v4.csv       标签量 → 性能（骨架分组 5 折 × 3 种子）
  feature_selection_v4.csv    重要性 Top-k → 性能（回答"多少特征就够"）
  feature_ranking_v4.csv      全特征重要性排序
"""
import os, json
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'
SEEDS = [0, 1, 42]

FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
LAB = pd.read_csv(f'{OUT}/labels_452.csv'); MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)
tr = ALL.merge(LAB[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner')
y = tr['dE水解_kJmol'].values
v = pd.read_csv('vlib_final.csv'); d1 = pd.read_csv('dataset_v1.csv')
smi = dict(zip(v['编号'], v['SMILES'])); smi.update(dict(zip(d1['编号'], d1['SMILES'])))
SMI = tr['编号'].map(smi).values

def scaf_of(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None: return 'bad'
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)
SCAF = np.array([scaf_of(s) for s in SMI]); UNIQ = np.unique(SCAF)

def folds_scaffold(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UNIQ))
    fo, sz = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sz)); fo[UNIQ[u]] = f; sz[f] += int(np.sum(SCAF == UNIQ[u]))
    return np.array([fo[x] for x in SCAF])

MK_FN = lambda s: RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4)

def cv(cols, seeds=SEEDS, sub_n=None):
    X = tr[cols].values; mae, r2s, rhos = [], [], []
    for s in seeds:
        folds = folds_scaffold(s)
        if sub_n is None:
            use = np.arange(len(y))
        else:
            use = np.random.RandomState(s).choice(len(y), min(sub_n, len(y)), replace=False)
        p = np.zeros(len(use)); yu = y[use]; fu = folds[use]
        # 折叠重映射（子集内）
        for f in range(5):
            te = np.where(fu == f)[0]; tri = np.where(fu != f)[0]
            if len(te) == 0 or len(tri) < 5: continue
            p[te] = MK_FN(s).fit(X[use][tri], yu[tri]).predict(X[use][te])
        mae.append(mean_absolute_error(yu, p)); r2s.append(r2_score(yu, p)); rhos.append(spearmanr(p, yu)[0])
    return round(float(np.mean(mae)),2), round(float(np.mean(r2s)),3), round(float(np.mean(rhos)),3)

# ---------- 学习曲线 ----------
print('=== 学习曲线 ===')
rows = []
for n in [30, 60, 100, 150, 200, 252, 302, 352, 402, 452]:
    mae, r2, rho = cv(BEST, sub_n=n)
    rows.append({'n_labels': n, 'MAE': mae, 'R2': r2, 'rho': rho})
    print(f'  n={n:4d}  MAE={mae:.2f}  R2={r2:+.3f}  rho={rho:.3f}')
pd.DataFrame(rows).to_csv(f'{OUT}/learning_curve_v4.csv', index=False)

# ---------- 特征重要性排序 + Top-k ----------
print('\n=== 特征选择（Top-k）===')
m = MK_FN(42).fit(tr[BEST].values, y)
imp = pd.DataFrame({'feature': BEST, 'importance': m.feature_importances_}).sort_values('importance', ascending=False)
imp['block'] = imp['feature'].apply(lambda c: 'mech' if c in MECH else 'base' if c in BASE else
                                    'morgan' if c.startswith('fp') else 'maccs')
imp.to_csv(f'{OUT}/feature_ranking_v4.csv', index=False)
print(imp.head(15).to_string(index=False))

rows = []
for k in [10, 20, 30, 46, 60, 100, 200, len(BEST)]:
    cols = imp['feature'].head(k).tolist()
    mae, r2, rho = cv(cols)
    rows.append({'top_k': k, 'MAE': mae, 'R2': r2, 'rho': rho,
                 '含机理块数': sum(1 for c in cols if c in MECH)})
    print(f'  top-{k:4d}  MAE={mae:.2f}  R2={r2:+.3f}  rho={rho:.3f}')
pd.DataFrame(rows).to_csv(f'{OUT}/feature_selection_v4.csv', index=False)

# ---------- 紧凑模型：仅机理块 ----------
print('\n=== 紧凑模型对比 ===')
comp = []
for nm, cols in [('base16', BASE), ('mech_only', MECH), ('base+mech', BASE + MECH),
                 ('morgan_only', FP), ('base+morgan', BASE + FP), ('full', BEST)]:
    mae, r2, rho = cv(cols)
    comp.append({'模型特征': nm, 'n': len(cols), 'MAE': mae, 'R2': r2, 'rho': rho})
    print(f'  {nm:14s} k={len(cols):4d}  MAE={mae:.2f}  R2={r2:+.3f}  rho={rho:.3f}')
pd.DataFrame(comp).to_csv(f'{OUT}/compact_model_table.csv', index=False)
print('\n完成。')
