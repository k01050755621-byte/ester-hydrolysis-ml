# -*- coding: utf-8 -*-
"""ML 补强 · 第 9 步：最终标签量（1099）下的稳健性与消融全套

输出：ml_final/final_ablation.csv / final_models.csv / final_butina.csv / final_yrandom.json
"""
import os, json
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'
SEEDS = [0, 1, 42]

FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
LAB = pd.read_csv(os.environ.get('LABFILE', f'{OUT}/labels_final.csv')); MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)
tr = ALL.merge(LAB[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner')
y = tr['dE水解_kJmol'].values
print('标签数', len(y))
v = pd.read_csv('vlib_final.csv'); d1 = pd.read_csv('dataset_v1.csv')
smi = dict(zip(v['编号'], v['SMILES'])); smi.update(dict(zip(d1['编号'], d1['SMILES'])))
SMI = tr['编号'].map(smi).values

def scaf_of(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None: return 'bad'
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)
SCAF = np.array([scaf_of(s) for s in SMI]); UNIQ = np.unique(SCAF)
def folds_scaf(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UNIQ)); fo, sz = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sz)); fo[UNIQ[u]] = f; sz[f] += int(np.sum(SCAF == UNIQ[u]))
    return np.array([fo[x] for x in SCAF])

BUF = {}
def folds_butina(seed):
    if 'cl' not in BUF:
        fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024) for s in SMI]
        n = len(fps); ds = []
        for i in range(1, n):
            ds.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
        BUF['cl'] = Butina.ClusterData(ds, n, 0.45, isDistData=True)
        print('  Butina 簇数', len(BUF['cl']))
    cl = BUF['cl']; cl_of = {mi: ci for ci, ms in enumerate(cl) for mi in ms}
    uq = np.unique(list(cl_of.values())); rng = np.random.RandomState(seed)
    order = rng.permutation(len(uq)); fo, sz = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sz)); fo[u] = f; sz[f] += sum(1 for x in cl_of.values() if x == u)
    return np.array([fo[cl_of[i]] for i in range(len(SMI))])

M = {
    'XGBoost': lambda s: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                      colsample_bytree=0.7, random_state=s, n_jobs=4),
    'RandomForest': lambda s: RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4),
    'ExtraTrees': lambda s: ExtraTreesRegressor(n_estimators=400, random_state=s, n_jobs=4),
    'GradBoost': lambda s: GradientBoostingRegressor(random_state=s),
    'HistGBDT': lambda s: HistGradientBoostingRegressor(random_state=s),
    'Ridge': lambda s: Ridge(alpha=1.0),
    'kNN5': lambda s: KNeighborsRegressor(n_neighbors=5),
    'SVR': lambda s: SVR(C=10.0, gamma='scale'),
}
def cv(cols, foldfn, mfn, seeds=SEEDS):
    X = tr[cols].values; mae, r2, rho, bmae = [], [], [], []
    for s in seeds:
        f = foldfn(s); p = np.zeros(len(y)); pb = np.zeros(len(y))
        for j in range(5):
            te, tri = np.where(f == j)[0], np.where(f != j)[0]
            p[te] = mfn(s).fit(X[tri], y[tri]).predict(X[te]); pb[te] = y[tri].mean()
        mae.append(mean_absolute_error(y, p)); r2.append(r2_score(y, p)); rho.append(spearmanr(p, y)[0])
        bmae.append(mean_absolute_error(y, pb))
    return {'MAE': round(float(np.mean(mae)), 2), 'MAE_std': round(float(np.std(mae)), 2),
            'R2': round(float(np.mean(r2)), 3), 'rho': round(float(np.mean(rho)), 3),
            'base_MAE': round(float(np.mean(bmae)), 2)}

print('\n=== 最终消融（1099 标签，骨架分组）===')
sets = {'base16': BASE, 'base+mech': BASE + MECH, 'base+FP': BASE + FP,
        'base+mech+FP': BASE + MECH + FP, 'full(+MACCS)': BEST, 'mech_only': MECH}
rows = []
for nm, cols in sets.items():
    r = cv(cols, folds_scaf, M['RandomForest'])
    rows.append({'特征集': nm, 'n_features': len(cols), **r})
    print(f'  {nm:16s} k={len(cols):4d}  MAE={r["MAE"]:.2f}  R2={r["R2"]:+.3f}  rho={r["rho"]:.3f}  '
          f'base={r["base_MAE"]:.2f}')
pd.DataFrame(rows).to_csv(f'{OUT}/final_ablation.csv', index=False)

print('\n=== 最终模型对比（1099 标签）===')
rows = []
for nm, fn in M.items():
    r = cv(BEST, folds_scaf, fn)
    rows.append({'模型': nm, **r}); print(f'  {nm:14s} MAE={r["MAE"]:.2f}  R2={r["R2"]:+.3f}  rho={r["rho"]:.3f}')
def ens(s):
    class E:
        def __init__(self, ss):
            self.ms = [M['XGBoost'](ss), M['RandomForest'](ss), M['HistGBDT'](ss)]
        def fit(self, X, yy):
            [m.fit(X, yy) for m in self.ms]; return self
        def predict(self, X): return np.mean([m.predict(X) for m in self.ms], axis=0)
    return E(s)
r = cv(BEST, folds_scaf, ens); rows.append({'模型': 'Ensemble(XGB+RF+Hist)', **r})
print(f'  Ensemble      MAE={r["MAE"]:.2f}  R2={r["R2"]:+.3f}  rho={r["rho"]:.3f}')
pd.DataFrame(rows).to_csv(f'{OUT}/final_models.csv', index=False)

print('\n=== 最终 Butina 分组 CV（更严）===')
rows = []
for nm, cols in {'base16': BASE, 'base+FP': BASE + FP, 'full(+MACCS)': BEST}.items():
    r = cv(cols, folds_butina, M['RandomForest'], seeds=[0, 1, 42])
    rows.append({'特征集': nm, **r})
    print(f'  {nm:14s} MAE={r["MAE"]:.2f}  R2={r["R2"]:+.3f}  rho={r["rho"]:.3f}  base={r["base_MAE"]:.2f}')
pd.DataFrame(rows).to_csv(f'{OUT}/final_butina.csv', index=False)

print('\n=== 最终 y-随机化（10 次）===')
yr = []
f = folds_scaf(42)
for s in range(10):
    yp = np.random.RandomState(1000 + s).permutation(y); p = np.zeros(len(yp))
    for j in range(5):
        te, tri = np.where(f == j)[0], np.where(f != j)[0]
        p[te] = M['RandomForest'](42).fit(tr[BEST].values[tri], yp[tri]).predict(tr[BEST].values[te])
    yr.append(mean_absolute_error(yp, p))
out = {'真实标签_MAE': cv(BEST, folds_scaf, M['RandomForest'], seeds=[42])['MAE'],
       'y随机化_MAE_mean': round(float(np.mean(yr)), 2), 'y随机化_MAE_std': round(float(np.std(yr)), 2),
       'y随机化_MAE_list': [round(float(x), 2) for x in yr]}
json.dump(out, open(f'{OUT}/final_yrandom.json', 'w'), ensure_ascii=False, indent=2)
print(' ', out)
print('\n完成。')
