# -*- coding: utf-8 -*-
"""ML 补强 · 第 4 步（v2 修正版）：最终模型 + conformal + 同分异构体分辨力

修正三处（方法学收紧）：
  1) 最终模型 = CV 前三名集成（RF + HistGBDT + XGB），剔除表现最差的 ExtraTrees；
  2) Mondrian conformal 的每类分位数 **只在校准集上估计**（此前误用了全部标签），并报告经验覆盖率；
  3) 补：真实覆盖率验证（校准集/测试集对半）。

输出：
  vlib_predictions_v4.csv / pareto_v4.csv / halogen_free_top10_v4.csv / all_negative_CI_v4.csv
  isomer_groups_v4.csv / isomer_collapse_summary.csv / isomer_within_group_rho.csv
  conformal_mondrian.json / key_molecules_v3_vs_v4.csv / final_model_summary.json
"""
import os, json
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'

FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
LAB = pd.read_csv(f'{OUT}/labels_452.csv'); MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)
tr = ALL.merge(LAB[['编号', 'dE水解_kJmol', '来源']], on='编号', how='inner')
y = tr['dE水解_kJmol'].values
Xtr = tr[BEST].values

# CV 前三名集成
def members(s):
    return [XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.7, random_state=s, n_jobs=4),
            RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4),
            HistGradientBoostingRegressor(random_state=s)]

def fit_predict(Xa, ya, Xb, seeds=(0, 1, 42, 7, 2024)):
    ps = []
    for s in seeds:
        for m in members(s):
            ps.append(m.fit(Xa, ya).predict(Xb))
    return np.mean(ps, axis=0)

def cls_of(row):
    return f"{'Ar' if row['n_acylAr'] > 0 else 'Al'}-{'Ar' if row['n_alkAr'] > 0 else 'Al'}"

# ---------- conformal：严格三分划（训练/校准/测试，按骨架分组防泄漏）----------
print('=== conformal（严格三分划：train / cal / test，骨架分组）===')
from rdkit.Chem.Scaffolds import MurckoScaffold
def scaf_of(s):
    m = Chem.MolFromSmiles(str(s))
    sc = MurckoScaffold.GetScaffoldForMol(m) if m is not None else None
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)
vmap = pd.read_csv('vlib_final.csv').set_index('编号')['SMILES'].to_dict()
d1map = pd.read_csv('dataset_v1.csv').set_index('编号')['SMILES'].to_dict()
SC = np.array([scaf_of(vmap.get(i, d1map.get(i, 'C'))) for i in tr['编号']])
UQ = np.unique(SC)

def split3(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UQ))
    target = np.array([0.5, 0.25, 0.25]) * len(tr); got = np.zeros(3); fo = {}
    for u in order:
        f = int(np.argmin(got / target))       # 相对进度最小 → 均衡
        fo[UQ[u]] = f; got[f] += int(np.sum(SC == UQ[u]))
    lab = np.array([fo[x] for x in SC])
    return np.where(lab == 0)[0], np.where(lab == 1)[0], np.where(lab == 2)[0]

tr_cls = tr.apply(cls_of, axis=1).values
QS, seed42 = [], {}
for sd in [42, 7, 2024]:
    tri, cal, tst = split3(sd)
    pc = fit_predict(Xtr[tri], y[tri], Xtr[cal], seeds=(42,))
    res_cal = np.abs(y[cal] - pc)
    q = float(np.quantile(res_cal, 0.9))
    pt = fit_predict(Xtr[tri], y[tri], Xtr[tst], seeds=(42,))
    res_tst = np.abs(y[tst] - pt)
    print(f'  split{ sd }: n_train={len(tri)} n_cal={len(cal)} n_test={len(tst)} | '
          f'q90={q:.2f} | 覆盖率={float((res_tst <= q).mean()):.1%}')
    QS.append({'seed': sd, 'n_train': len(tri), 'n_cal': len(cal), 'n_test': len(tst),
               'q90': round(q, 2), 'coverage_test': round(float((res_tst <= q).mean()), 3)})
    if sd == 42:
        seed42 = {'tri': tri, 'cal': cal, 'tst': tst, 'res_cal': res_cal, 'res_tst': res_tst, 'q90': q}
q90 = float(np.mean([x['q90'] for x in QS]))
q80 = float(np.quantile(seed42['res_cal'], 0.8))
cov_tst = float(np.mean([x['coverage_test'] for x in QS]))
print(f'  汇总：q90 = {q90:.2f} | 测试集覆盖率（名义 90%）= {cov_tst:.1%}')

cal, tst = seed42['cal'], seed42['tst']; res_cal = seed42['res_cal']; res_tst = seed42['res_tst']
mond = {}; cov_m = {}
for c in sorted(set(tr_cls)):
    mc_cal = (tr_cls == c)[cal]; mc_tst = (tr_cls == c)[tst]
    if mc_cal.sum() < 20:
        mond[c] = {'n_cal': int(mc_cal.sum()), 'q90': round(q90, 2), 'note': '样本不足，退全局 q90'}
        cov_m[c] = None; continue
    qc = float(np.quantile(res_cal[mc_cal], 0.9))
    mond[c] = {'n_cal': int(mc_cal.sum()), 'q90': round(qc, 2),
               'q80': round(float(np.quantile(res_cal[mc_cal], 0.8)), 2)}
    cov_m[c] = round(float((res_tst[mc_tst] <= qc).mean()), 3) if mc_tst.sum() else None
print(' ', json.dumps(mond, ensure_ascii=False))
print('  各类测试集覆盖率:', json.dumps(cov_m, ensure_ascii=False))

# ---------- 最终模型 → 全库 ----------
print('\n=== 最终模型（RF+HistGBDT+XGB, 5 种子）===')
pred_lib = fit_predict(Xtr, y, FV[BEST].values)
vsrc = pd.read_csv('vlib_final.csv')
V = vsrc[['编号', '名称', 'SMILES', '分子量']].copy()
V['预测_v4'] = pred_lib
FV_cls = FV.apply(cls_of, axis=1).values
V['q90_v4'] = [mond[c]['q90'] for c in FV_cls]
V['CI下_v4'] = V['预测_v4'] - V['q90_v4']; V['CI上_v4'] = V['预测_v4'] + V['q90_v4']

srt = V.sort_values(['预测_v4', '分子量'], ascending=[True, False])
front, bmw = [], -1
for i, rr in srt.iterrows():
    if rr['分子量'] > bmw:
        front.append(i); bmw = rr['分子量']
V['Pareto_v4'] = False; V.loc[front, 'Pareto_v4'] = True

def nhal(s):
    m = Chem.MolFromSmiles(s)
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() in ('F', 'Cl', 'Br', 'I'))
V['n_hal'] = [nhal(s) for s in V['SMILES']]
nBr = V['SMILES'].str.count('Br'); nCl = V['SMILES'].str.count('Cl')
V['rot'] = [Lipinski.NumRotatableBonds(Chem.MolFromSmiles(s)) for s in V['SMILES']]
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
V['总分_v4'] = z(-V['预测_v4']) + 0.5*z(V['分子量']) - z(nBr + 0.5*nCl) - 0.5*z(V['rot'])
V['_idn'] = V['编号'].str[1:].astype(int)
hf = V[V['n_hal'] == 0].sort_values(['总分_v4', '预测_v4', '分子量', '_idn'],
                                    ascending=[False, True, False, True]).reset_index(drop=True)
hf['排名_v4'] = np.arange(1, len(hf)+1)
print(f'  不同预测值 {V["预测_v4"].round(4).nunique()}/{len(V)} | 区间全负 {int((V["CI上_v4"]<0).sum())} | '
      f'Pareto {int(V["Pareto_v4"].sum())} | 真无卤 {len(hf)}')

V[['编号','名称','SMILES','分子量','预测_v4','q90_v4','CI下_v4','CI上_v4','Pareto_v4','n_hal']].to_csv(
    f'{OUT}/vlib_predictions_v4.csv', index=False)
hf.head(10)[['编号','名称','预测_v4','分子量','总分_v4','排名_v4','n_hal']].to_csv(
    f'{OUT}/halogen_free_top10_v4.csv', index=False)
V[V['Pareto_v4']][['编号','名称','预测_v4','分子量','CI下_v4','CI上_v4']].to_csv(f'{OUT}/pareto_v4.csv', index=False)
V[V['CI上_v4'] < 0][['编号','名称','预测_v4','CI下_v4','CI上_v4','分子量']].sort_values('预测_v4').to_csv(
    f'{OUT}/all_negative_CI_v4.csv', index=False)

# ---------- 同分异构体分辨力 ----------
print('\n=== 同分异构体分辨力 ===')
key = FV[BASE].round(6).astype(str).agg('|'.join, axis=1).values
V['_bk'] = key
m_base = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.7, random_state=42, n_jobs=4).fit(tr[BASE].values, y)
V['pred_base'] = m_base.predict(FV[BASE].values)
rows = []
for bk, sub in V.groupby('_bk'):
    if len(sub) < 2: continue
    rows.append({'组大小': len(sub), 'base不同预测数': sub['pred_base'].round(4).nunique(),
                 'v4不同预测数': sub['预测_v4'].round(4).nunique()})
ig = pd.DataFrame(rows); ig.to_csv(f'{OUT}/isomer_groups_v4.csv', index=False)
print(f'  含 ≥2 同描述符分子的组数 = {len(ig)}，覆盖 {int(ig["组大小"].sum())} 个分子')
print(f'  组内平均不同预测数：base16 = {ig["base不同预测数"].mean():.3f} | v4 = {ig["v4不同预测数"].mean():.3f}')
ig_all = []
for nm, p in [('base16', V['pred_base'].values), ('v4', V['预测_v4'].values)]:
    nd = len(np.unique(np.round(p, 4)))
    ig_all.append({'模型': nm, '不同预测值数': int(nd), '塌缩比': round(len(V)/nd, 3)})
pd.DataFrame(ig_all).to_csv(f'{OUT}/isomer_collapse_summary.csv', index=False)
print(pd.DataFrame(ig_all).to_string(index=False))

lab_map = dict(zip(LAB['编号'], LAB['dE水解_kJmol']))
V['label'] = V['编号'].map(lab_map)
wrows = []
for bk, sub in V.groupby('_bk'):
    s = sub.dropna(subset=['label'])
    if len(s) < 3: continue
    rb = spearmanr(s['pred_base'], s['label'])[0]
    rv = spearmanr(s['预测_v4'], s['label'])[0]
    wrows.append({'组大小': len(s), 'rho_base': rb, 'rho_v4': rv})
wd = pd.DataFrame(wrows)
n_def = int(wd['rho_base'].notna().sum())
print(f'\n  有 ≥3 个带标签同描述符分子的组数 = {len(wd)}（其中 base 预测非常数、ρ 可定义的有 {n_def} 组）')
if len(wd):
    print(f'  组内 ρ：base = {np.nanmean(wd["rho_base"]):.3f}（{n_def} 组可定义） | v4 = {np.nanmean(wd["rho_v4"]):.3f}')
wd.to_csv(f'{OUT}/isomer_within_group_rho.csv', index=False)

# ---------- 关键分子 ----------
v3 = pd.read_csv(f'{OUT}/../vlib_predictions_v3.csv'); v3p = dict(zip(v3['编号'], v3['预测_v3']))
rows = []
for cid in ['C1074','C6533','C12225','C8188','C2841','C820','C3486','C6181','C9162','C9360','C1223','C9132','C2606']:
    r = V[V['编号'] == cid]
    if len(r) == 0: continue
    rows.append({'编号': cid, 'xTB标签': lab_map.get(cid), 'v3预测': round(float(v3p[cid]),1) if cid in v3p else None,
                 'v4预测': round(float(r['预测_v4'].iloc[0]),1),
                 'v4_CI上': round(float(r['CI上_v4'].iloc[0]),1), 'v4_CI下': round(float(r['CI下_v4'].iloc[0]),1),
                 '真无卤': bool(r['n_hal'].iloc[0] == 0)})
kd = pd.DataFrame(rows); kd.to_csv(f'{OUT}/key_molecules_v3_vs_v4.csv', index=False)
print('\n关键分子对照:'); print(kd.to_string(index=False))

json.dump({'final_model': 'Ensemble(RF+HistGBDT+XGB, 5 seeds)', 'q90_global': round(q90,2),
           'q80_global': round(q80,2), 'coverage_test_global': round(cov_tst,3),
           'mondrian': mond, 'mondrian_coverage': cov_m,
           'n_distinct_pred': int(V['预测_v4'].round(4).nunique()),
           'n_CI_all_negative': int((V['CI上_v4'] < 0).sum()),
           'n_pareto': int(V['Pareto_v4'].sum()), 'n_halogen_free_strict': int(len(hf)),
           'isomer_collapse': ig_all,
           'isomer_group_mean_distinct': {'base': round(float(ig['base不同预测数'].mean()),3),
                                          'v4': round(float(ig['v4不同预测数'].mean()),3)},
           'halogen_free_top10': hf.head(10)['编号'].tolist(),
           'pareto_members': V[V['Pareto_v4']]['编号'].tolist()},
          open(f'{OUT}/final_model_summary.json','w'), ensure_ascii=False, indent=2, default=str)
print('\n完成。')
