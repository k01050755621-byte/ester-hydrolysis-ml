# -*- coding: utf-8 -*-
"""ML 补强 · 第 7 步：并入 AL4 标签 → v5 模型（标签量 452 → 772）

内容：
  ① 并入 al4_labels.csv，形成 772 标签集；
  ② v4 模型对 AL4 320 个全新分子的**前瞻验证**（最高等级证据）；
  ③ 452 vs 772 的骨架分组 CV 对照；
  ④ v5 全库预测 + Pareto + 真无卤榜 + 区间全负；
  ⑤ v4 vs v5 关键分子对照。

输出：ml_final/ 下的 labels_772.csv / v5_* / prospective_v4_on_AL4.csv 等
"""
import os, json
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

# ---------- 标签合并 ----------
a3 = pd.read_csv('al_round3_labels.csv'); a3 = a3[a3['dE水解_kJmol'].notna()]
a4 = pd.read_csv(f'{OUT}/al4_labels.csv'); a4 = a4[a4['dE水解_kJmol'].notna()]
lab452 = pd.read_csv(f'{OUT}/labels_452.csv')
lab = pd.concat([lab452[['编号', 'dE水解_kJmol', '来源']],
                 a4[['编号', 'dE水解_kJmol']].assign(来源='AL4')], ignore_index=True)
lab = lab.drop_duplicates('编号', keep='first')
tr = ALL.merge(lab[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner')
print(f'标签总数 {len(tr)} | 302 {(tr["来源"]=="302").sum()} | AL3 {(tr["来源"]=="AL3").sum()} | '
      f'AL4 {(tr["来源"]=="AL4").sum()}')
y = tr['dE水解_kJmol'].values; X = tr[BEST].values
lab.to_csv(f'{OUT}/labels_772.csv', index=False)

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

# ---------- ① 452 vs 772 的骨架分组 CV ----------
print('\n=== 标签量对照（骨架分组 5 折 × 3 种子）===')
cv = {}
for tag, mask in [('452', (tr['来源'] != 'AL4').values), ('772', np.ones(len(tr), bool))]:
    mae, r2, rho, bmae = [], [], [], []
    for s in [0, 1, 42]:
        f = folds_of(s); Xm, ym, fm = X[mask], y[mask], f[mask]; p = np.zeros(len(ym)); pb = np.zeros(len(ym))
        for j in range(5):
            te, tri = np.where(fm == j)[0], np.where(fm != j)[0]
            pr = np.mean([m.fit(Xm[tri], ym[tri]).predict(Xm[te]) for m in MODELS(s)], axis=0)
            p[te] = pr; pb[te] = ym[tri].mean()
        mae.append(mean_absolute_error(ym, p)); r2.append(r2_score(ym, p)); rho.append(spearmanr(p, ym)[0])
        bmae.append(mean_absolute_error(ym, pb))
    cv[tag] = {'n': int(mask.sum()), 'MAE': round(float(np.mean(mae)), 2), 'R2': round(float(np.mean(r2)), 3),
               'rho': round(float(np.mean(rho)), 3), 'base_MAE': round(float(np.mean(bmae)), 2),
               'R2_std': round(float(np.std(r2)), 3)}
    print(f'  n={cv[tag]["n"]:4d}  MAE={cv[tag]["MAE"]:.2f}  R2={cv[tag]["R2"]:+.3f}  '
          f'rho={cv[tag]["rho"]:.3f}  base={cv[tag]["base_MAE"]:.2f}')

# ---------- ② 前瞻验证：v4 模型对 AL4 320 个全新分子 ----------
print('\n=== 前瞻验证（v4 模型 → AL4 新标签）===')
v4 = pd.read_csv(f'{OUT}/vlib_predictions_v4.csv'); v4p = dict(zip(v4['编号'], v4['预测_v4']))
ok = a4.dropna(subset=['dE水解_kJmol']).copy(); ok['v4预测'] = ok['编号'].map(v4p)
ok = ok.dropna(subset=['v4预测'])
rho, pv = spearmanr(ok['v4预测'], ok['dE水解_kJmol'])
prosp = {'n': int(len(ok)), 'Spearman': round(float(rho), 3), 'p': float(pv),
         'MAE': round(float((ok['v4预测'] - ok['dE水解_kJmol']).abs().mean()), 2),
         'R2': round(float(r2_score(ok['dE水解_kJmol'], ok['v4预测'])), 3)}
for t, g in ok.groupby('query类型'):
    prosp[f'{t}_n'] = int(len(g)); prosp[f'{t}_MAE'] = round(float((g['v4预测'] - g['dE水解_kJmol']).abs().mean()), 2)
    if len(g) > 2:
        prosp[f'{t}_Spearman'] = round(float(spearmanr(g['v4预测'], g['dE水解_kJmol'])[0]), 3)
print(' ', json.dumps(prosp, ensure_ascii=False))
ok.to_csv(f'{OUT}/prospective_v4_on_AL4.csv', index=False)

# ---------- ③ v5 模型 → 全库 ----------
print('\n=== v5 最终模型 → 全库 ===')
def fit_pred(Xa, ya, Xb, seeds=(0, 1, 42, 7, 2024)):
    return np.mean([m.fit(Xa, ya).predict(Xb) for s in seeds for m in MODELS(s)], axis=0)
pred = fit_pred(X, y, FV[BEST].values)
V = pd.read_csv('vlib_final.csv')[['编号', '名称', 'SMILES', '分子量']].copy()
V['预测_v5'] = pred
# conformal（三分划，骨架分组）
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
    QS.append({'q90': q, 'coverage': float((rt <= q).mean())})
q90 = float(np.mean([x['q90'] for x in QS])); cov = float(np.mean([x['coverage'] for x in QS]))
print(f'  q90 = {q90:.2f} | 测试集覆盖率（名义 90%）= {cov:.1%}')
V['q90_v5'] = q90; V['CI下_v5'] = V['预测_v5'] - q90; V['CI上_v5'] = V['预测_v5'] + q90

srt = V.sort_values(['预测_v5', '分子量'], ascending=[True, False]); front, bmw = [], -1
for i, rr in srt.iterrows():
    if rr['分子量'] > bmw: front.append(i); bmw = rr['分子量']
V['Pareto_v5'] = False; V.loc[front, 'Pareto_v5'] = True
def nhal(s):
    m = Chem.MolFromSmiles(s); return sum(1 for a in m.GetAtoms() if a.GetSymbol() in ('F','Cl','Br','I'))
V['n_hal'] = [nhal(s) for s in V['SMILES']]
nBr = V['SMILES'].str.count('Br'); nCl = V['SMILES'].str.count('Cl')
V['rot'] = [Lipinski.NumRotatableBonds(Chem.MolFromSmiles(s)) for s in V['SMILES']]
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
V['总分_v5'] = z(-V['预测_v5']) + .5*z(V['分子量']) - z(nBr + .5*nCl) - .5*z(V['rot'])
V['_idn'] = V['编号'].str[1:].astype(int)
hf = V[V['n_hal'] == 0].sort_values(['总分_v5', '预测_v5', '分子量', '_idn'],
                                    ascending=[False, True, False, True]).reset_index(drop=True)
hf['排名_v5'] = np.arange(1, len(hf)+1)
print(f'  不同预测值 {V["预测_v5"].round(4).nunique()}/{len(V)} | 区间全负 {int((V["CI上_v5"]<0).sum())} | '
      f'Pareto {int(V["Pareto_v5"].sum())} | 真无卤 {len(hf)}')
V[['编号','名称','SMILES','分子量','预测_v5','q90_v5','CI下_v5','CI上_v5','Pareto_v5','n_hal']].to_csv(
    f'{OUT}/vlib_predictions_v5.csv', index=False)
hf.head(10)[['编号','名称','预测_v5','分子量','总分_v5','排名_v5','n_hal']].to_csv(
    f'{OUT}/halogen_free_top10_v5.csv', index=False)
V[V['Pareto_v5']][['编号','名称','预测_v5','分子量','CI下_v5','CI上_v5']].to_csv(f'{OUT}/pareto_v5.csv', index=False)
V[V['CI上_v5'] < 0][['编号','名称','预测_v5','CI下_v5','CI上_v5','分子量']].sort_values('预测_v5').to_csv(
    f'{OUT}/all_negative_CI_v5.csv', index=False)

# ---------- ④ 关键分子 v4 vs v5 ----------
lm = dict(zip(lab['编号'], lab['dE水解_kJmol']))
rows = []
for cid in ['C1074','C6533','C12225','C8188','C2841','C820','C3486','C6181','C9162','C9360','C1223','C9132','C2606']:
    r = V[V['编号'] == cid]
    if len(r) == 0: continue
    rows.append({'编号': cid, 'xTB标签': lm.get(cid), 'v4预测': round(float(v4p.get(cid, np.nan)),1),
                 'v5预测': round(float(r['预测_v5'].iloc[0]),1),
                 'v5_CI上': round(float(r['CI上_v5'].iloc[0]),1), 'v5_CI下': round(float(r['CI下_v5'].iloc[0]),1)})
kd = pd.DataFrame(rows); kd.to_csv(f'{OUT}/key_molecules_v4_vs_v5.csv', index=False)
print('\n关键分子对照:'); print(kd.to_string(index=False))

json.dump({'labels': int(len(tr)), 'cv': cv, 'prospective_v4_on_AL4': prosp,
           'q90': round(q90,2), 'coverage_test': round(cov,3),
           'n_distinct_pred': int(V['预测_v5'].round(4).nunique()),
           'n_CI_all_negative': int((V['CI上_v5']<0).sum()),
           'pareto_members': V[V['Pareto_v5']]['编号'].tolist(),
           'halogen_free_top10': hf.head(10)['编号'].tolist()},
          open(f'{OUT}/v5_summary.json','w'), ensure_ascii=False, indent=2, default=str)
print('\n完成。')
