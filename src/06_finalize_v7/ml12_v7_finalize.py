# -*- coding: utf-8 -*-
"""ML 补强 · 第 12 步：v7 口径的全套最终收尾

v7（多构象去噪标签）已是主标签集，但下游还有几样东西停在 v6 口径：
  1) 逐轮前瞻验证（v6_summary.json 里的 AL3/AL4/AL5 是 v6 标签算的）；
  2) 榜单四件套（all_negative_CI / pareto / halogen_free_top10 / key_molecules）停在 v6；
  3) 没有一份把 v7 全部数字集中起来的汇总。

本脚本用 v7 标签重算以上全部，产出 v7_final_summary.json。
不改任何标签文件，只读 + 写新产物。

用法： python ml12_v7_finalize.py
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
MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)

LAB = pd.read_csv(f'{OUT}/labels_final_v7.csv')          # 编号, dE水解_kJmol(=v7), 来源
tr = ALL.merge(LAB, on='编号', how='inner')
y = tr['dE水解_kJmol'].values; X = tr[BEST].values
print(f'v7 标签总数 {len(tr)}')
print(tr['来源'].value_counts().to_string())

MODELS = lambda s: [XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.7, random_state=s, n_jobs=4),
                    RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4),
                    HistGradientBoostingRegressor(random_state=s)]

# ================= 1) 逐轮前瞻验证（v7 口径）=================
print('\n=== 逐轮前瞻验证（v7 标签，训练集只用该轮之前的标签）===')
PROS = {}
ROUNDS = [('AL3', {'302'}, 'AL3'), ('AL4', {'302', 'AL3'}, 'AL4'), ('AL5', {'302', 'AL3', 'AL4'}, 'AL5')]
for tag, tr_src, te_src in ROUNDS:
    mtr = tr['来源'].isin(tr_src).values; mte = (tr['来源'] == te_src).values
    if mtr.sum() == 0 or mte.sum() == 0:
        print(f'  {tag}: 跳过（训练 {mtr.sum()} / 测试 {mte.sum()}）'); continue
    Xa, ya = X[mtr], y[mtr]; Xb, yb = X[mte], y[mte]
    pred = np.mean([m.fit(Xa, ya).predict(Xb) for s in (0, 1, 42) for m in MODELS(s)], axis=0)
    rho, pv = spearmanr(pred, yb)
    PROS[tag] = {'n_train': int(mtr.sum()), 'n': int(mte.sum()),
                 'Spearman': round(float(rho), 3), 'p': float(pv),
                 'MAE': round(float(mean_absolute_error(yb, pred)), 2),
                 'R2': round(float(r2_score(yb, pred)), 3),
                 'base_MAE': round(float(np.abs(yb - ya.mean()).mean()), 2)}
    print(f'  {tag}: 训练 {mtr.sum():4d} -> 预测 {mte.sum():3d} 个新分子  ρ={rho:.3f}  '
          f'MAE={PROS[tag]["MAE"]:.2f}  R²={PROS[tag]["R2"]:+.3f}  均值基线={PROS[tag]["base_MAE"]:.2f}  p={pv:.1e}')

# ================= 2) 榜单四件套（v7）=================
print('\n=== v7 榜单重排 ===')
VP = pd.read_csv(f'{OUT}/vlib_predictions_v7.csv')
VP6 = pd.read_csv(f'{OUT}/vlib_predictions_v6.csv')[['编号', '预测_v6']]
V = pd.read_csv('vlib_final.csv')[['编号', '名称', 'SMILES', '分子量']].merge(
    VP[['编号', '预测_v7', 'q90_v7', 'CI下_v7', 'CI上_v7', 'n_hal']], on='编号', how='left').merge(
    VP6, on='编号', how='left')
V = V.rename(columns={'预测_v7': '预测_v7'})
srt = V.sort_values(['预测_v7', '分子量'], ascending=[True, False]); front, bmw = [], -1
for i, rr in srt.iterrows():
    if rr['分子量'] > bmw: front.append(i); bmw = rr['分子量']
V['Pareto_v7'] = False; V.loc[front, 'Pareto_v7'] = True

nBr = V['SMILES'].str.count('Br'); nCl = V['SMILES'].str.count('Cl')
V['rot'] = [Lipinski.NumRotatableBonds(Chem.MolFromSmiles(s)) for s in V['SMILES']]
z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
V['总分_v7'] = z(-V['预测_v7']) + .5 * z(V['分子量']) - z(nBr + .5 * nCl) - .5 * z(V['rot'])
V['_idn'] = V['编号'].str[1:].astype(int)
hf = V[V['n_hal'] == 0].sort_values(['总分_v7', '预测_v7', '分子量', '_idn'],
                                    ascending=[False, True, False, True]).reset_index(drop=True)
hf['排名_v7'] = np.arange(1, len(hf) + 1)

allneg = V[V['CI上_v7'] < 0][['编号', '名称', '预测_v7', 'CI下_v7', 'CI上_v7', '分子量']].sort_values('预测_v7')
pareto = V[V['Pareto_v7']][['编号', '名称', '预测_v7', '分子量', 'CI下_v7', 'CI上_v7', 'n_hal']]
allneg.to_csv(f'{OUT}/all_negative_CI_v7.csv', index=False)
pareto.to_csv(f'{OUT}/pareto_v7.csv', index=False)
hf.head(10)[['编号', '名称', '预测_v7', '分子量', '总分_v7', '排名_v7', 'n_hal']].to_csv(
    f'{OUT}/halogen_free_top10_v7.csv', index=False)

print(f'  不同预测值 {V["预测_v7"].round(4).nunique()}/{len(V)} | 90%区间全负 {len(allneg)} | '
      f'Pareto {len(pareto)} | 真无卤池 {len(hf)}')
print('\n  区间全负名单（前 20）:'); print(allneg.head(20).to_string(index=False))
print('\n  Pareto 前沿:'); print(pareto.to_string(index=False))
print('\n  真无卤 Top-10:'); print(hf.head(10)[['编号', '名称', '预测_v7', '分子量', '排名_v7']].to_string(index=False))

# ================= 3) 关键分子 v7 对照 ==================
lm = dict(zip(LAB['编号'], LAB['dE水解_kJmol']))
lm_old = dict(zip(pd.read_csv(f'{OUT}/labels_v7_final.csv')['编号'],
                  pd.read_csv(f'{OUT}/labels_v7_final.csv')['dE_old']))
v4p = dict(zip(pd.read_csv(f'{OUT}/vlib_predictions_v4.csv')['编号'],
               pd.read_csv(f'{OUT}/vlib_predictions_v4.csv')['预测_v4']))
v5p = dict(zip(pd.read_csv(f'{OUT}/vlib_predictions_v5.csv')['编号'],
               pd.read_csv(f'{OUT}/vlib_predictions_v5.csv')['预测_v5']))
v6p = dict(zip(pd.read_csv(f'{OUT}/vlib_predictions_v6.csv')['编号'],
               pd.read_csv(f'{OUT}/vlib_predictions_v6.csv')['预测_v6']))
rows = []
for cid in ['C1074', 'C6533', 'C12225', 'C8188', 'C2841', 'C820', 'C3486', 'C6181',
            'C9162', 'C9360', 'C1223', 'C9132', 'C2606', 'C6488']:
    r = V[V['编号'] == cid]
    if len(r) == 0: continue
    r = r.iloc[0]
    rows.append({'编号': cid,
                 'xTB_v7': (round(float(lm[cid]), 1) if cid in lm else None),
                 'xTB_v6': (round(float(lm_old[cid]), 1) if cid in lm_old else None),
                 'v4预测': round(v4p.get(cid, float('nan')), 1),
                 'v5预测': round(v5p.get(cid, float('nan')), 1),
                 'v6预测': round(v6p.get(cid, float('nan')), 1),
                 'v7预测': round(float(r['预测_v7']), 1),
                 'v7_CI下': round(float(r['CI下_v7']), 1),
                 'v7_CI上': round(float(r['CI上_v7']), 1)})
kd = pd.DataFrame(rows); kd.to_csv(f'{OUT}/key_molecules_v7.csv', index=False)
print('\n关键分子 v7 对照:'); print(kd.to_string(index=False))

# ================= 4) 预测分布 v6 vs v7 ==================
print('\n=== 预测分布对照（v6 模型 vs v7 模型，同一库）===')
DIS = []
for nm, col, q in [('v6', '预测_v6', 15.31), ('v7', '预测_v7', 14.53)]:
    p = V[col].values
    DIS.append({'版本': nm, 'q90': q, 'mean': round(float(p.mean()), 2), 'sd': round(float(p.std()), 2),
                'min': round(float(p.min()), 1), 'max': round(float(p.max()), 1),
                'n_lt_-20': int((p < -20).sum()), 'n_lt_-15': int((p < -15).sum()),
                'n_ge_5': int((p >= 5).sum())})
    print(f'  {nm}: mean={DIS[-1]["mean"]:+.2f} sd={DIS[-1]["sd"]:.2f} min={DIS[-1]["min"]:.1f} '
          f'| pred<-20: {DIS[-1]["n_lt_-20"]} | <-15: {DIS[-1]["n_lt_-15"]}')

# ================= 5) 汇总 ==================
v6s = json.load(open(f'{OUT}/v6_summary.json'))
v7l = json.load(open(f'{OUT}/v7_summary.json'))
iso = json.load(open(f'{OUT}/isomer_final_summary.json'))
dr = json.load(open(f'{OUT}/design_rules_summary.json'))
fin_models = pd.read_csv(f'{OUT}/final_models.csv').to_dict('records')
fin_abl = pd.read_csv(f'{OUT}/final_ablation.csv').to_dict('records')
fin_but = pd.read_csv(f'{OUT}/final_butina.csv').to_dict('records')
yr = json.load(open(f'{OUT}/final_yrandom.json'))

json.dump({'label_set': 'labels_final_v7.csv (multi-conformer GFN2, 6 confs, fixed seed)',
           'n_labels': int(len(tr)), 'source_counts': tr['来源'].value_counts().to_dict(),
           'label_ladder': v7l['ladder_v7'], 'label_ladder_v6': v7l['ladder_v6'],
           'prospective': PROS,
           'conformal': {'q90': v7l['q90'], 'coverage_test': v7l['coverage_test'],
                         'splits': v7l['conformal_splits']},
           'butina': v7l['butina'], 'yrandom': yr,
           'models': fin_models, 'ablation': fin_abl, 'butina_featsets': fin_but,
           'n_distinct_pred': int(V['预测_v7'].round(4).nunique()),
           'n_CI_all_negative': int(len(allneg)), 'n_pareto': int(len(pareto)),
           'pareto_members': pareto['编号'].tolist(),
           'halogen_free_top10': hf.head(10)['编号'].tolist(),
           'n_halogen_free_pool': int(len(hf)),
           'isomer': iso,
           'label_diff_v6_v7': v7l['label_diff'],
           'lib_v6_v7_spearman': v7l['lib_v6_v7_spearman'],
           'dist_compare': DIS,
           'design_rules': {'crosstab': pd.read_csv(f'{OUT}/rule_crosstab.csv').to_dict('records'),
                            'halogen_bins': pd.read_csv(f'{OUT}/rule_halogen_bins.csv').to_dict('records'),
                            'univariate': pd.read_csv(f'{OUT}/rule_univariate.csv').to_dict('records'),
                            'linear_model': dr['linear_model'],
                            'shap_top20': dr['top20'][:12],
                            'block_share': {k: dr[k] for k in ('base', 'mech', 'morgan', 'maccs')}}},
          open(f'{OUT}/v7_final_summary.json', 'w'), ensure_ascii=False, indent=2, default=str)
print('\n完成 -> v7_final_summary.json / all_negative_CI_v7.csv / pareto_v7.csv / '
      'halogen_free_top10_v7.csv / key_molecules_v7.csv')
