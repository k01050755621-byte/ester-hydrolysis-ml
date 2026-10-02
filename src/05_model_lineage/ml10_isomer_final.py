# -*- coding: utf-8 -*-
"""ML 补强 · 第 10 步：最终标签集（1099）下的同分异构体分辨力（核心图数据）

输出：ml_final/isomer_groups_final.csv / isomer_collapse_final.csv / isomer_within_group_rho_final.csv
"""
import os, json
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'
FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE = MAN['base_cols']
LAB = pd.read_csv(os.environ.get('LABFILE', f'{OUT}/labels_final.csv'))
# 口径修复：全特征臂必须与 base16 臂**同时**切到当前标签版本对应的预测文件，
# 否则会出现"base16 用新标签重训、全特征却仍是旧模型"的混淆比较。
PREDF = os.environ.get('PREDFILE', f'{OUT}/vlib_predictions_v7.csv')
PREDC = os.environ.get('PREDCOL', '预测_v7')
V6 = pd.read_csv(PREDF)
print(f'[口径] 标签 = {os.path.basename(os.environ.get("LABFILE", "labels_final.csv"))} | '
      f'全特征预测 = {os.path.basename(PREDF)}:{PREDC}')
# 用最终标签集在 base16 特征上训练一个对照模型
FL = pd.read_csv(f'{OUT}/feat_lit.csv')
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)
tr = ALL.merge(LAB[['编号', 'dE水解_kJmol']], on='编号', how='inner')
m_base = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.7, random_state=42, n_jobs=4).fit(tr[BASE].values, tr['dE水解_kJmol'].values)
V = FV[['编号']].copy()
V['pred_base'] = m_base.predict(FV[BASE].values)
V['pred_full'] = V['编号'].map(dict(zip(V6['编号'], V6[PREDC])))
V['_bk'] = FV[BASE].round(6).astype(str).agg('|'.join, axis=1).values
V['label'] = V['编号'].map(dict(zip(LAB['编号'], LAB['dE水解_kJmol'])))

rows = []
for bk, sub in V.groupby('_bk'):
    if len(sub) < 2: continue
    rows.append({'组大小': len(sub), 'base不同预测数': sub['pred_base'].round(4).nunique(),
                 'v6不同预测数': sub['pred_full'].round(4).nunique()})
ig = pd.DataFrame(rows); ig.to_csv(f'{OUT}/isomer_groups_final.csv', index=False)
summ = []
for nm, p in [('base16', V['pred_base'].values), ('v6', V['pred_full'].values)]:
    nd = len(np.unique(np.round(p, 4)))
    summ.append({'模型': nm, '不同预测值数': int(nd), '塌缩比': round(len(V) / nd, 3)})
sd = pd.DataFrame(summ); sd.to_csv(f'{OUT}/isomer_collapse_final.csv', index=False)
print(f'含 ≥2 同描述符分子的组数 = {len(ig)}，覆盖 {int(ig["组大小"].sum())} 个分子')
print(f'组内平均不同预测数：base16 = {ig["base不同预测数"].mean():.3f} | v6 = {ig["v6不同预测数"].mean():.3f}')
print(sd.to_string(index=False))

wr = []
for bk, sub in V.groupby('_bk'):
    s = sub.dropna(subset=['label'])
    if len(s) < 3: continue
    rb = spearmanr(s['pred_base'], s['label'])[0]
    rv = spearmanr(s['pred_full'], s['label'])[0]
    wr.append({'组大小': len(s), 'rho_base': rb, 'rho_v6': rv})
wd = pd.DataFrame(wr); wd.to_csv(f'{OUT}/isomer_within_group_rho_final.csv', index=False)
nd = int(wd['rho_base'].notna().sum())
print(f'\n有 ≥3 个带标签同描述符分子的组数 = {len(wd)}（base 预测为常数、ρ 可定义的组数 = {nd}）')
print(f'组内 ρ：base16 = {np.nanmean(wd["rho_base"]):.3f} | 全特征 = {np.nanmean(wd["rho_v6"]):.3f}')
# 按组大小分层（小样本组会主导均值，必须分开报）
print('  组大小分层（全特征臂）：')
strat = []
for lo, hi, nm in [(3, 3, 'n=3'), (4, 7, 'n=4-7'), (8, 10 ** 9, 'n>=8')]:
    m = wd[(wd['组大小'] >= lo) & (wd['组大小'] <= hi)]
    if len(m):
        strat.append({'层': nm, '组数': int(len(m)), '中位rho': round(float(m['rho_v6'].median()), 3),
                      '均值rho': round(float(m['rho_v6'].mean()), 3),
                      '加权rho': round(float(np.average(m['rho_v6'], weights=m['组大小'])), 3)})
        print(f'    {nm:6s} 组数={len(m):3d}  中位ρ={m["rho_v6"].median():+.3f}  均值ρ={m["rho_v6"].mean():+.3f}')
json.dump({'n_groups': int(len(ig)), 'n_covered': int(ig['组大小'].sum()),
           'mean_distinct_base': round(float(ig['base不同预测数'].mean()), 3),
           'mean_distinct_v6': round(float(ig['v6不同预测数'].mean()), 3),
           'collapse': sd.to_dict('records'),
           'within_group_n': int(len(wd)), 'within_group_rho_base': None if nd == 0 else round(float(np.nanmean(wd['rho_base'])), 3),
           'within_group_rho_v6': round(float(np.nanmean(wd['rho_v6'])), 3),
           'within_group_median_rho_full': round(float(wd['rho_v6'].median()), 3),
           'within_group_strata': strat,
           'label_file': os.path.basename(os.environ.get('LABFILE', 'labels_final.csv')),
           'pred_file': os.path.basename(PREDF)},
          open(f'{OUT}/isomer_final_summary.json', 'w'), ensure_ascii=False, indent=2)
print('完成。')
