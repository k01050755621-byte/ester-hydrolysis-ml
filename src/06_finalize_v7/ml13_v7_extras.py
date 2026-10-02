# -*- coding: utf-8 -*-
"""ML 补强 · 第 13 步：v7 口径补齐三处残留（报告定稿用）

背景：v6 报告里有三块数字来自更早的轮次，换成 v7 标签后不成立，必须重算：
  1) Mondrian conformal：原来自 ml4_final_model.py，是 452 标签 / v4 预测的口径。
  2) 学习曲线（池内子采样 vs 逐轮 AL）：原来自 ml5_learn_select.py，标签是 labels_452.csv。
  3) Butina 簇数：报告写 130/132 不一致，需要一次权威计算。

输出：
  mondrian_v7.csv / mondrian_v7.json
  learning_curve_v7.csv
  butina_clusters_v7.json
"""
import os, json
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'

FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
LAB = pd.read_csv(f'{OUT}/labels_final_v7.csv')          # ★ v7 标签
MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)
tr = ALL.merge(LAB[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner').reset_index(drop=True)
y = tr['dE水解_kJmol'].values
SRC = tr['来源'].values
print(f'标签数 {len(y)}（来源 {pd.Series(SRC).value_counts().to_dict()}）| 特征 {len(BEST)}')
# 自检：静默丢数据防线
assert len(tr) == len(LAB), f'合并后行数 {len(tr)} != 标签数 {len(LAB)}'
print(f'[check] 输入 {len(LAB)} 行 -> 训练表 {len(tr)} 行  (OK)')

# ---------------- 骨架分组 ----------------
v = pd.read_csv('vlib_final.csv'); d1 = pd.read_csv('dataset_v1.csv')
vmap = dict(zip(v['编号'], v['SMILES'])); d1map = dict(zip(d1['编号'], d1['SMILES']))
def scaf_of(s):
    m = Chem.MolFromSmiles(str(s))
    sc = MurckoScaffold.GetScaffoldForMol(m) if m is not None else None
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)
miss = [i for i in tr['编号'] if i not in vmap and i not in d1map]
print(f'[check] 取不到 SMILES 的编号数 = {len(miss)}')
SC = np.array([scaf_of(vmap.get(i, d1map.get(i, 'C'))) for i in tr['编号']])
UQ = np.unique(SC)
def folds_of(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UQ)); fo, sz = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sz)); fo[UQ[u]] = f; sz[f] += int(np.sum(SC == UQ[u]))
    return np.array([fo[x] for x in SC])

def MK_FN(s): return RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4)

# ---------------- 1) Mondrian conformal（v7 口径，类别 = 酰基侧/烷氧基侧是否芳环）----------------
def cls_of(row): return f"{'Ar' if row['n_acylAr'] > 0 else 'Al'}-{'Ar' if row['n_alkAr'] > 0 else 'Al'}"
CLS = tr.apply(cls_of, axis=1).values
Xtr = tr[BEST].values

def split3(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UQ))
    tgt = np.array([.5, .25, .25]) * len(tr); got = np.zeros(3); fo = {}
    for u in order:
        f = int(np.argmin(got / tgt)); fo[UQ[u]] = f; got[f] += int(np.sum(SC == UQ[u]))
    lb = np.array([fo[x] for x in SC]); return np.where(lb == 0)[0], np.where(lb == 1)[0], np.where(lb == 2)[0]

def fit_predict(Xa, ya, Xb, seeds=(42,)):
    ps = []
    for s in seeds:
        for m in (XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                               colsample_bytree=0.7, random_state=s, n_jobs=4),
                  RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4),
                  HistGradientBoostingRegressor(random_state=s)):
            ps.append(m.fit(Xa, ya).predict(Xb))
    return np.mean(ps, axis=0)

print('\n=== Mondrian conformal（v7，严格三分划，骨架分组）===')
QS, seed42 = [], None
for sd in [42, 7, 2024]:
    tri, cal, tst = split3(sd)
    rc = np.abs(y[cal] - fit_predict(Xtr[tri], y[tri], Xtr[cal]))
    q = float(np.quantile(rc, .9))
    rt = np.abs(y[tst] - fit_predict(Xtr[tri], y[tri], Xtr[tst]))
    print(f'  split{sd}: n_train={len(tri)} n_cal={len(cal)} n_test={len(tst)} | q90={q:.2f} | '
          f'覆盖率={float((rt<=q).mean()):.1%}')
    QS.append({'seed': sd, 'q90': round(q, 2), 'coverage_test': round(float((rt <= q).mean()), 3)})
    if sd == 42:
        seed42 = {'tri': tri, 'cal': cal, 'tst': tst, 'res_cal': rc, 'res_tst': rt, 'q90': q}

q90g = float(np.mean([x['q90'] for x in QS])); covg = float(np.mean([x['coverage_test'] for x in QS]))
print(f'  汇总：q90 = {q90g:.2f} | 测试集覆盖率（名义 90%）= {covg:.1%}')

cal, tst = seed42['cal'], seed42['tst']; rc, rt = seed42['res_cal'], seed42['res_tst']
mond, cov_m, rows = {}, {}, []
for c in sorted(set(CLS)):
    mc, mt = (CLS == c)[cal], (CLS == c)[tst]
    if mc.sum() < 20:
        mond[c] = {'n_cal': int(mc.sum()), 'q90': round(q90g, 2), 'note': '样本不足，退全局 q90'}
        cov_m[c] = None; continue
    qc = float(np.quantile(rc[mc], .9))
    cvg = float((rt[mt] <= qc).mean()) if mt.sum() else None
    mond[c] = {'n_cal': int(mc.sum()), 'n_test': int(mt.sum()), 'q90': round(qc, 2), '宽度_95CI': round(2 * qc, 2)}
    cov_m[c] = None if cvg is None else round(cvg, 3)
    rows.append({'类别': c, 'n_校准': int(mc.sum()), 'n_测试': int(mt.sum()), 'q90': round(qc, 2),
                 '区间宽度_90%': round(2 * qc, 2), '实测覆盖率': cov_m[c],
                 '标签均值': round(float(y[cal][mc].mean()), 2)})
    print(f'  {c}: n_cal={int(mc.sum())} q90={qc:.2f} 宽度={2*qc:.1f} 覆盖率={cov_m[c]}')
pd.DataFrame(rows).to_csv(f'{OUT}/mondrian_v7.csv', index=False)
json.dump({'q90_global': round(q90g, 2), 'coverage_test_global': round(covg, 3),
           'splits': QS, 'mondrian': mond, 'mondrian_coverage': cov_m,
           'label_file': 'labels_final_v7.csv'},
          open(f'{OUT}/mondrian_v7.json', 'w'), ensure_ascii=False, indent=2)

# ---------------- 2) 学习曲线（v7）：池内子采样 vs 逐轮 AL ----------------
print('\n=== 学习曲线（v7 标签）===')
POOL = np.where(np.isin(SRC, ['302', 'AL3']))[0]          # 452 池
print(f'  452 池 = {len(POOL)}')
def cv_sub(use, seeds=(0, 1, 42)):
    mae, r2s, rhos = [], [], []
    for s in seeds:
        f = folds_of(s); X = Xtr[use]; yu = y[use]; fu = f[use]
        p = np.zeros(len(use))
        for j in range(5):
            te, tri = np.where(fu == j)[0], np.where(fu != j)[0]
            if len(te) == 0 or len(tri) < 5: continue
            p[te] = MK_FN(s).fit(X[tri], yu[tri]).predict(X[te])
        mae.append(mean_absolute_error(yu, p)); r2s.append(r2_score(yu, p)); rhos.append(spearmanr(p, yu)[0])
    return round(float(np.mean(mae)), 2), round(float(np.mean(r2s)), 3), round(float(np.mean(rhos)), 3)

rows = []
for n in [30, 60, 100, 150, 200, 252, 302, 352, 402, 452]:
    mae, r2, rho = cv_sub(np.random.RandomState(0).choice(POOL, n, replace=False))
    rows.append({'n_labels': n, 'MAE': mae, 'R2': r2, 'rho': rho, '来源': '池内子采样'})
    print(f'  [子采样] n={n:4d}  MAE={mae:.2f}  R2={r2:+.3f}  rho={rho:.3f}')
for n, srcs in [(452, ['302', 'AL3']), (771, ['302', 'AL3', 'AL4']), (1099, ['302', 'AL3', 'AL4', 'AL5'])]:
    use = np.where(np.isin(SRC, srcs))[0]
    mae, r2, rho = cv_sub(use)
    rows.append({'n_labels': n, 'MAE': mae, 'R2': r2, 'rho': rho, '来源': '逐轮 AL（真新增化学空间）'})
    print(f'  [AL]     n={n:4d}  MAE={mae:.2f}  R2={r2:+.3f}  rho={rho:.3f}')
pd.DataFrame(rows).to_csv(f'{OUT}/learning_curve_v7.csv', index=False)

# ---------------- 3) Butina 簇数（权威，1099 训练集）----------------
print('\n=== Butina 聚类（1099 训练集，Tanimoto 阈值 0.45）===')
fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(vmap.get(i, d1map.get(i, 'C'))), 2, nBits=1024)
       for i in tr['编号']]
n = len(fps); ds = []
for i in range(1, n):
    ds.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
cl = Butina.ClusterData(ds, n, 0.45, isDistData=True)
ncl = len(cl); maxc = max(len(c) for c in cl)
print(f'  簇数 = {ncl} | 最大簇 = {maxc} | 平均簇大小 = {n/ncl:.2f}')
json.dump({'n_molecules': n, 'threshold': 0.45, 'n_clusters': ncl, 'max_cluster': maxc,
           'mean_cluster': round(n / ncl, 2)}, open(f'{OUT}/butina_clusters_v7.json', 'w'),
          ensure_ascii=False, indent=2)
print('\n完成 -> mondrian_v7.csv/.json, learning_curve_v7.csv, butina_clusters_v7.json')
