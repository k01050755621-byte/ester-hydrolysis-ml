# -*- coding: utf-8 -*-
"""ML 补强 · 第 8 步：并入 AL4 + AL5 → v6 模型（最终版）

输出：ml_final/ 下 labels_final.csv / v6_summary.json / vlib_predictions_v6.csv 等
另加：标签量阶梯对照（302 / 452 / 771 / ~1100）与逐轮前瞻验证汇总。
"""
import os, json, glob
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'

FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)

# ---------- 合并全部标签 ----------
lab = pd.read_csv(f'{OUT}/labels_772.csv')
extra = []
for rnd in [4, 5]:
    f = f'{OUT}/al{rnd}_labels.csv'
    if os.path.exists(f):
        d = pd.read_csv(f); d = d[d['dE水解_kJmol'].notna()]
        extra.append(d[['编号', 'dE水解_kJmol']].assign(来源=f'AL{rnd}'))
lab = pd.concat([lab] + extra, ignore_index=True).drop_duplicates('编号', keep='first')
tr = ALL.merge(lab[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner')
y = tr['dE水解_kJmol'].values; X = tr[BEST].values
print('标签总数', len(tr)); print(tr['来源'].value_counts().to_string())
lab.to_csv(f'{OUT}/labels_final.csv', index=False)

vmap = pd.read_csv('vlib_final.csv').set_index('编号')['SMILES'].to_dict()
d1map = pd.read_csv('dataset_v1.csv').set_index('编号')['SMILES'].to_dict()
def scaf_of(s):
    m = Chem.MolFromSmiles(str(s)); sc = MurckoScaffold.GetScaffoldForMol(m) if m is not None else None
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)
SC = np.array([scaf_of(vmap.get(i, d1map.get(i, 'C'))) for i in tr['编号']]); UQ = np.unique(SC)
def folds_of(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UQ)); fo, sz = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sz)); fo[UQ[u]] = f; sz[f] += int(np.sum(SC == UQ[u]))
    return np.array([fo[x] for x in SC])
MODELS = lambda s: [XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.7, random_state=s, n_jobs=4),
                    RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4),
                    HistGradientBoostingRegressor(random_state=s)]

# ---------- 标签量阶梯 ----------
print('\n=== 标签量阶梯（骨架分组 5 折 × 3 种子）===')
stages = [('302', ['302']), ('452', ['302', 'AL3']), ('771', ['302', 'AL3', 'AL4']),
          ('final', list(tr['来源'].unique()))]
cv = {}
for tag, srcs in stages:
    mask = tr['来源'].isin(srcs).values
    mae, r2, rho, bmae = [], [], [], []
    for s in [0, 1, 42]:
        f = folds_of(s); Xm, ym, fm = X[mask], y[mask], f[mask]
        p = np.zeros(len(ym)); pb = np.zeros(len(ym))
        for j in range(5):
            te, tri = np.where(fm == j)[0], np.where(fm != j)[0]
            p[te] = np.mean([m.fit(Xm[tri], ym[tri]).predict(Xm[te]) for m in MODELS(s)], axis=0)
            pb[te] = ym[tri].mean()
        mae.append(mean_absolute_error(ym, p)); r2.append(r2_score(ym, p)); rho.append(spearmanr(p, ym)[0])
        bmae.append(mean_absolute_error(ym, pb))
    cv[tag] = {'n': int(mask.sum()), 'MAE': round(float(np.mean(mae)), 2), 'R2': round(float(np.mean(r2)), 3),
               'rho': round(float(np.mean(rho)), 3), 'base_MAE': round(float(np.mean(bmae)), 2)}
    print(f'  {tag:6s} n={cv[tag]["n"]:4d}  MAE={cv[tag]["MAE"]:.2f}  R2={cv[tag]["R2"]:+.3f}  '
          f'rho={cv[tag]["rho"]:.3f}  base={cv[tag]["base_MAE"]:.2f}')
pd.DataFrame(cv).T.to_csv(f'{OUT}/label_ladder.csv')

# ---------- 逐轮前瞻验证 ----------
print('\n=== 逐轮前瞻验证 ===')
prosp = {}
ROUNDS = [(3, f'{OUT}/al3_labels.csv', '../vlib_predictions_retrained.csv', '预测_新模型'),
          (4, f'{OUT}/al4_labels.csv', f'{OUT}/vlib_predictions_v4.csv', '预测_v4'),
          (5, f'{OUT}/al5_labels.csv', f'{OUT}/vlib_predictions_v5.csv', '预测_v5')]
if not os.path.exists(ROUNDS[0][1]):
    ROUNDS[0] = (3, 'al_round3_labels.csv', f'{OUT}/../vlib_predictions_retrained.csv', '预测_新模型')
for rnd, labf, predf, col in ROUNDS:
    if not os.path.exists(labf) or not os.path.exists(predf):
        print(f'  AL{rnd}: 跳过（缺文件 {labf if not os.path.exists(labf) else predf}）'); continue
    d = pd.read_csv(labf); d = d[d['dE水解_kJmol'].notna()]
    pf = pd.read_csv(predf)
    pmap = dict(zip(pf['编号'], pf[col]))
    d = d.assign(pred=d['编号'].map(pmap)).dropna(subset=['pred'])
    rho, pv = spearmanr(d['pred'], d['dE水解_kJmol'])
    prosp[f'AL{rnd}'] = {'n': int(len(d)), 'Spearman': round(float(rho), 3), 'p': float(pv),
                         'MAE': round(float((d['pred'] - d['dE水解_kJmol']).abs().mean()), 2),
                         'R2': round(float(r2_score(d['dE水解_kJmol'], d['pred'])), 3)}
    print(f'  AL{rnd}: n={len(d)}  rho={rho:.3f}  MAE={prosp[f"AL{rnd}"]["MAE"]:.2f}  '
          f'R2={prosp[f"AL{rnd}"]["R2"]:+.3f}  p={pv:.1e}')

# ---------- 最终模型 → 全库 ----------
print('\n=== v6 最终模型 → 全库 ===')
def fit_pred(Xa, ya, Xb, seeds=(0, 1, 42, 7, 2024)):
    return np.mean([m.fit(Xa, ya).predict(Xb) for s in seeds for m in MODELS(s)], axis=0)
pred = fit_pred(X, y, FV[BEST].values)
V = pd.read_csv('vlib_final.csv')[['编号', '名称', 'SMILES', '分子量']].copy()
V['预测_v6'] = pred
def split3(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UQ))
    tgt = np.array([.5, .25, .25]) * len(tr); got = np.zeros(3); fo = {}
    for u in order:
        f = int(np.argmin(got / tgt)); fo[UQ[u]] = f; got[f] += int(np.sum(SC == UQ[u]))
    lb = np.array([fo[x] for x in SC]); return np.where(lb == 0)[0], np.where(lb == 1)[0], np.where(lb == 2)[0]
QS = []
for sd in [42, 7, 2024]:
    tri, cal, tst = split3(sd)
    rc = np.abs(y[cal] - fit_pred(X[tri], y[tri], X[cal], seeds=(42,)))
    q = float(np.quantile(rc, .9)); rt = np.abs(y[tst] - fit_pred(X[tri], y[tri], X[tst], seeds=(42,)))
    QS.append({'q90': round(q, 2), 'coverage': round(float((rt <= q).mean()), 3)})
q90 = float(np.mean([x['q90'] for x in QS])); cov = float(np.mean([x['coverage'] for x in QS]))
print(f'  q90 = {q90:.2f} | 覆盖率（名义 90%）= {cov:.1%} | 各划分 {QS}')
V['q90_v6'] = q90; V['CI下_v6'] = V['预测_v6'] - q90; V['CI上_v6'] = V['预测_v6'] + q90
srt = V.sort_values(['预测_v6', '分子量'], ascending=[True, False]); front, bmw = [], -1
for i, rr in srt.iterrows():
    if rr['分子量'] > bmw: front.append(i); bmw = rr['分子量']
V['Pareto_v6'] = False; V.loc[front, 'Pareto_v6'] = True
def nhal(s):
    m = Chem.MolFromSmiles(s); return sum(1 for a in m.GetAtoms() if a.GetSymbol() in ('F','Cl','Br','I'))
V['n_hal'] = [nhal(s) for s in V['SMILES']]
nBr = V['SMILES'].str.count('Br'); nCl = V['SMILES'].str.count('Cl')
V['rot'] = [Lipinski.NumRotatableBonds(Chem.MolFromSmiles(s)) for s in V['SMILES']]
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
V['总分_v6'] = z(-V['预测_v6']) + .5*z(V['分子量']) - z(nBr + .5*nCl) - .5*z(V['rot'])
V['_idn'] = V['编号'].str[1:].astype(int)
hf = V[V['n_hal'] == 0].sort_values(['总分_v6', '预测_v6', '分子量', '_idn'],
                                    ascending=[False, True, False, True]).reset_index(drop=True)
hf['排名_v6'] = np.arange(1, len(hf)+1)
print(f'  不同预测值 {V["预测_v6"].round(4).nunique()}/{len(V)} | 区间全负 {int((V["CI上_v6"]<0).sum())} | '
      f'Pareto {int(V["Pareto_v6"].sum())} | 真无卤 {len(hf)}')
V[['编号','名称','SMILES','分子量','预测_v6','q90_v6','CI下_v6','CI上_v6','Pareto_v6','n_hal']].to_csv(
    f'{OUT}/vlib_predictions_v6.csv', index=False)
hf.head(10)[['编号','名称','预测_v6','分子量','总分_v6','排名_v6','n_hal']].to_csv(
    f'{OUT}/halogen_free_top10_v6.csv', index=False)
V[V['Pareto_v6']][['编号','名称','预测_v6','分子量','CI下_v6','CI上_v6']].to_csv(f'{OUT}/pareto_v6.csv', index=False)
V[V['CI上_v6'] < 0][['编号','名称','预测_v6','CI下_v6','CI上_v6','分子量']].sort_values('预测_v6').to_csv(
    f'{OUT}/all_negative_CI_v6.csv', index=False)

lm = dict(zip(lab['编号'], lab['dE水解_kJmol']))
v4p = dict(zip(pd.read_csv(f'{OUT}/vlib_predictions_v4.csv')['编号'],
               pd.read_csv(f'{OUT}/vlib_predictions_v4.csv')['预测_v4']))
v5p = dict(zip(pd.read_csv(f'{OUT}/vlib_predictions_v5.csv')['编号'],
               pd.read_csv(f'{OUT}/vlib_predictions_v5.csv')['预测_v5']))
rows = []
for cid in ['C1074','C6533','C12225','C8188','C2841','C820','C3486','C6181','C9162','C9360',
            'C1223','C9132','C2606','C12']:
    r = V[V['编号'] == cid]
    if len(r) == 0: continue
    rows.append({'编号': cid, 'xTB标签': lm.get(cid), 'v4预测': round(v4p.get(cid, float('nan')), 1),
                 'v5预测': round(v5p.get(cid, float('nan')), 1),
                 'v6预测': round(float(r['预测_v6'].iloc[0]), 1),
                 'v6_CI上': round(float(r['CI上_v6'].iloc[0]), 1),
                 'v6_CI下': round(float(r['CI下_v6'].iloc[0]), 1)})
kd = pd.DataFrame(rows); kd.to_csv(f'{OUT}/key_molecules_final.csv', index=False)
print('\n关键分子对照:'); print(kd.to_string(index=False))
json.dump({'n_labels': int(len(tr)), 'source_counts': tr['来源'].value_counts().to_dict(),
           'label_ladder': cv, 'prospective': prosp, 'q90': round(q90, 2),
           'coverage_test': round(cov, 3), 'conformal_splits': QS,
           'n_distinct_pred': int(V['预测_v6'].round(4).nunique()),
           'n_CI_all_negative': int((V['CI上_v6'] < 0).sum()),
           'n_pareto': int(V['Pareto_v6'].sum()),
           'pareto_members': V[V['Pareto_v6']]['编号'].tolist(),
           'halogen_free_top10': hf.head(10)['编号'].tolist()},
          open(f'{OUT}/v6_summary.json', 'w'), ensure_ascii=False, indent=2, default=str)
print('\n完成。')
