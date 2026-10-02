# -*- coding: utf-8 -*-
"""ML 补强 · 第 2 步：消融 / 模型对比 / 稳健性检验

协议（全部同折对照域内均值基线）：
  骨架分组 5 折 × 5 种子；另加 Butina 相似性分组 5 折（更严）。

输出：
  ablation_table.csv        特征块消融（MAE/R²/ρ/多样性/异构成因）
  model_table.csv           模型对比
  yrandom_table.json        y-随机化（证明非泄漏）
  robustness_summary.json
"""
import os, json, itertools
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'
SEEDS = [0, 1, 42, 7, 2024]

FV = pd.read_csv(f'{OUT}/feat_vlib.csv')
FL = pd.read_csv(f'{OUT}/feat_lit.csv')
LAB = pd.read_csv(f'{OUT}/labels_452.csv')
MAN = json.load(open(f'{OUT}/feature_manifest.json'))

BASE = MAN['base_cols']; MECH = MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp')]
MK = [c for c in FV.columns if c.startswith('mk')]
# 只保留在库中非零出现的位（其余为常量，无信息）
FP = [c for c in FP if FV[c].sum() > 0]
MK = [c for c in MK if FV[c].sum() > 0]

# ---------- 合并特征与标签 ----------
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)
tr = ALL.merge(LAB[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner')
assert len(tr) == len(LAB), (len(tr), len(LAB))
v = pd.read_csv('vlib_final.csv'); d1 = pd.read_csv('dataset_v1.csv')
smi = dict(zip(v['编号'], v['SMILES'])); smi.update(dict(zip(d1['编号'], d1['SMILES'])))
SMI = tr['编号'].map(smi).values
assert all(isinstance(s, str) for s in SMI), 'SMILES 缺失'
y = tr['dE水解_kJmol'].values


def scaf_of(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None: return 'bad'
    sc = MurckoScaffold.GetScaffoldForMol(m)
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)


SCAF = np.array([scaf_of(s) for s in SMI])
UNIQ = np.unique(SCAF)


def folds_scaffold(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UNIQ))
    fold_of, sizes = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sizes)); fold_of[UNIQ[u]] = f; sizes[f] += int(np.sum(SCAF == UNIQ[u]))
    return np.array([fold_of[x] for x in SCAF])


def folds_butina(seed):
    """Butina 相似性聚类分组（阈值 0.45 Tanimoto），按簇分配折。"""
    ms = [Chem.MolFromSmiles(s) for s in SMI]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in ms]
    n = len(fps); dists = []
    for i in range(1, n):
        for j in range(i):
            dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    cl = Butina.ClusterData(dists, n, 0.45, isDistData=True)
    cl_of = {}
    for ci, members in enumerate(cl):
        for mi in members: cl_of[mi] = ci
    uniq = np.unique(list(cl_of.values()))
    rng = np.random.RandomState(seed); order = rng.permutation(len(uniq))
    fold_of, sizes = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sizes)); fold_of[u] = f
        sizes[f] += sum(1 for x in cl_of.values() if x == u)
    return np.array([fold_of[cl_of[i]] for i in range(n)])


MODELS = {
    'XGBoost': lambda s: XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.9, colsample_bytree=0.7, random_state=s, n_jobs=4),
    'RandomForest': lambda s: RandomForestRegressor(n_estimators=400, max_depth=None, random_state=s, n_jobs=4),
    'ExtraTrees': lambda s: ExtraTreesRegressor(n_estimators=400, random_state=s, n_jobs=4),
    'GradBoost': lambda s: GradientBoostingRegressor(random_state=s),
    'HistGBDT': lambda s: HistGradientBoostingRegressor(random_state=s),
    'Ridge': lambda s: Ridge(alpha=1.0),
    'kNN5': lambda s: KNeighborsRegressor(n_neighbors=5),
    'SVR': lambda s: SVR(C=10.0, gamma='scale'),
}


def cv_eval(cols, foldfn, model_fn, seeds=SEEDS):
    X = tr[cols].values
    mae_l, r2_l, rho_l, base_l = [], [], [], []
    for s in seeds:
        folds = foldfn(s)
        p = np.zeros(len(y))
        for f in range(5):
            te, tri = np.where(folds == f)[0], np.where(folds != f)[0]
            p[te] = model_fn(s).fit(X[tri], y[tri]).predict(X[te])
        mae_l.append(mean_absolute_error(y, p)); r2_l.append(r2_score(y, p))
        rho_l.append(spearmanr(p, y)[0])
        # 域内均值基线（同折）
        pb = np.zeros(len(y))
        for f in range(5):
            te, tri = np.where(folds == f)[0], np.where(folds != f)[0]
            pb[te] = y[tri].mean()
        base_l.append(mean_absolute_error(y, pb))
    return {'MAE': round(float(np.mean(mae_l)), 2), 'MAE_std': round(float(np.std(mae_l)), 2),
            'R2': round(float(np.mean(r2_l)), 3), 'R2_std': round(float(np.std(r2_l)), 3),
            'rho': round(float(np.mean(rho_l)), 3),
            'base_MAE': round(float(np.mean(base_l)), 2)}


def diversity(cols, model_fn):
    """在全库 12,380 上预测，度量预测多样性 + 同分异构塌缩。"""
    m = model_fn(42).fit(tr[cols].values, y)
    pv = m.predict(FV[cols].values)
    n_dist = len(np.unique(np.round(pv, 4)))
    # 塌缩比：分子数 / 不同预测值数
    return {'n_distinct': int(n_dist), 'collapse_ratio': round(len(pv) / n_dist, 3)}


# ================= 消融 =================
print('=== 1. 特征块消融（骨架分组 5 折 × 5 种子）===')
FOLD_SETS = {
    'base_only': BASE,
    'base+mech': BASE + MECH,
    'base+fp(v3口径)': BASE + FP,
    'base+mech+fp': BASE + MECH + FP,
    'base+mech+fp+maccs': BASE + MECH + FP + MK,
}
abl = []
for name, cols in FOLD_SETS.items():
    r = cv_eval(cols, folds_scaffold, MODELS['XGBoost'])
    dv = diversity(cols, MODELS['XGBoost'])
    row = {'特征集': name, 'n_features': len(cols), **r, **dv}
    abl.append(row)
    print(f'  {name:22s} k={len(cols):4d}  MAE={r["MAE"]:.2f}  R2={r["R2"]:+.3f}  '
          f'rho={r["rho"]:.3f}  base={r["base_MAE"]:.2f}  不同预测值={dv["n_distinct"]}')
abl_df = pd.DataFrame(abl); abl_df.to_csv(f'{OUT}/ablation_table.csv', index=False)

# ================= 模型对比（最优特征集）=================
BEST = BASE + MECH + FP + MK
print('\n=== 2. 模型对比（最优特征集, 骨架分组）===')
mrows = []
for mname, fn in MODELS.items():
    r = cv_eval(BEST, folds_scaffold, fn)
    mrows.append({'模型': mname, **r})
    print(f'  {mname:14s} MAE={r["MAE"]:.2f}  R2={r["R2"]:+.3f}  rho={r["rho"]:.3f}  (base {r["base_MAE"]:.2f})')
# 简单集成：XGB+RF+ET 平均
def ens(s):
    class E:
        def __init__(self, ss):
            self.ms = [MODELS['XGBoost'](ss), MODELS['RandomForest'](ss), MODELS['ExtraTrees'](ss)]
        def fit(self, X, yy):
            for m in self.ms: m.fit(X, yy)
            return self
        def predict(self, X):
            return np.mean([m.predict(X) for m in self.ms], axis=0)
    return E(s)
r = cv_eval(BEST, folds_scaffold, ens)
mrows.append({'模型': 'Ensemble(XGB+RF+ET)', **r})
print(f'  Ensemble(XGB+RF+ET) MAE={r["MAE"]:.2f}  R2={r["R2"]:+.3f}  rho={r["rho"]:.3f}')
pd.DataFrame(mrows).to_csv(f'{OUT}/model_table.csv', index=False)

# ================= 更严协议：Butina 相似性分组 =================
print('\n=== 3. Butina 相似性分组（更严）===')
str_row = []
for name, cols in [('base_only', BASE), ('base+fp(v3口径)', BASE + FP), ('best', BEST)]:
    r = cv_eval(cols, folds_butina, MODELS['XGBoost'], seeds=[0, 1, 42])
    str_row.append({'特征集': name, **r})
    print(f'  {name:22s} MAE={r["MAE"]:.2f}  R2={r["R2"]:+.3f}  rho={r["rho"]:.3f}  base={r["base_MAE"]:.2f}')
pd.DataFrame(str_row).to_csv(f'{OUT}/butina_group_cv.csv', index=False)

# ================= y-随机化 =================
print('\n=== 4. y-随机化（10 次）===')
yr = []
for s in range(10):
    rng = np.random.RandomState(1000 + s)
    y_perm = rng.permutation(y)
    folds = folds_scaffold(42); p = np.zeros(len(y_perm))
    for f in range(5):
        te, tri = np.where(folds == f)[0], np.where(folds != f)[0]
        p[te] = MODELS['XGBoost'](42).fit(tr[BEST].values[tri], y_perm[tri]).predict(tr[BEST].values[te])
    yr.append(mean_absolute_error(y_perm, p))
yr_json = {'真实标签_MAE': cv_eval(BEST, folds_scaffold, MODELS['XGBoost'], seeds=[42])['MAE'],
           'y随机化_MAE_mean': round(float(np.mean(yr)), 2),
           'y随机化_MAE_std': round(float(np.std(yr)), 2),
           'y随机化_MAE_list': [round(float(x), 2) for x in yr]}
json.dump(yr_json, open(f'{OUT}/yrandom_table.json', 'w'), ensure_ascii=False, indent=2)
print(' ', yr_json)

json.dump({'ablation': abl, 'models': mrows, 'butina': str_row, 'yrandom': yr_json},
          open(f'{OUT}/robustness_summary.json', 'w'), ensure_ascii=False, indent=2, default=str)
print('\n完成。')
