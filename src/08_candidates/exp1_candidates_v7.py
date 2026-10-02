# -*- coding: utf-8 -*-
"""Exp-1 候选表 v7 版：把排序测定实验的候选清单从 v6 口径升到 v7（多构象去噪标签 + v7 模型）。

原则（沿用 v3 定案，不变）：
  1) 候选必须**横跨预测谱**（禁止在 Top 榜内部互比）——模型跨化学空间 ρ≈0.80，窄邻域只有 0.05–0.16；
  2) 排序组优先选**无标签**分子（真盲测）；已标注的只作锚点并如实标注；
  3) C1074 不进排序组（留作第四篇材料母体）。

与 v6 版的差别：
  - 预测/区间全部换成 v7（q90 = 14.53）；
  - xTB 标签列改用 **v7 标签**（同时保留 v6 值便于看新旧差）；
  - 排序组的选取理由按 v7 实际数值重写。

输出：
  exp1_candidate_tiers_v7.csv   全库分层候选（强/中/弱）
  exp1_candidate_tiers.csv      同内容覆盖（保持活动文档一致）
  exp1_ranking_set_v7.csv       推荐排序测定组（5 支，跨档）
"""
import os
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
ML = 'fix_20260928/ml_final'

V = pd.read_csv(f'{ML}/vlib_predictions_v7.csv')                       # 全库 v7 预测
L7 = pd.read_csv(f'{ML}/labels_final_v7.csv').rename(columns={'dE水解_kJmol': 'xTB标签_v7'})
L6 = pd.read_csv(f'{ML}/labels_v7_final.csv')[['编号', 'dE_old']].rename(columns={'dE_old': 'xTB标签_v6'})
V = V.merge(L7[['编号', 'xTB标签_v7', '来源']], on='编号', how='left').merge(L6, on='编号', how='left')

V['无卤'] = V['n_hal'] == 0
V['smi_len'] = V['SMILES'].str.len()
V['已标注'] = V['xTB标签_v7'].notna()

print('库内分子', len(V), '| 无卤', int(V['无卤'].sum()),
      '| 无卤且未标注', int((V['无卤'] & ~V['已标注']).sum()))
print('90% 区间全负:', int((V['CI上_v7'] < 0).sum()))
print('预测值不同数:', V['预测_v7'].round(4).nunique(), '/', len(V))

cols = ['编号', '名称', '分子量', '预测_v7', 'CI下_v7', 'CI上_v7',
        'xTB标签_v7', '已标注', '无卤', 'n_hal', 'smi_len']

strong = V[V['CI上_v7'] < 0].sort_values('CI上_v7')
med = V[(V['预测_v7'].between(-15, -5)) & (V['CI下_v7'] < 0) & (V['CI上_v7'] > 0)
        & V['无卤'] & ~V['已标注']].sort_values('smi_len')
weak = V[(V['预测_v7'] >= 5) & V['无卤'] & ~V['已标注']].sort_values('smi_len')

for nm, d in [('强', strong), ('中', med), ('弱', weak)]:
    print(f'\n{nm}档 n={len(d)}')
    print(d[cols].head(8).to_string(index=False))

out = pd.concat([strong.assign(档位='强'), med.assign(档位='中'), weak.assign(档位='弱')],
                ignore_index=True).sort_values(['档位', '预测_v7'])
out.to_csv('fix_20260928/exp1_candidate_tiers_v7.csv', index=False)
out.to_csv('fix_20260928/exp1_candidate_tiers.csv', index=False)
print('\nsaved -> exp1_candidate_tiers_v7.csv (+覆盖 exp1_candidate_tiers.csv)')

# ================= 推荐排序测定组：跨谱 5 支 =================
print('\n===== 推荐排序组（跨预测谱，C1074 已排除）=====')
SET = [
    ('C2001',  '强负',   '强负锚点（已标注）。v7 全库预测最负（−25.3），标签 −33.5；含 1 个 Br'),
    ('C12225', '负',     '无卤强负锚点（已标注）。S–S 二硫酯，v7 标签 −31.5 为全库最负，模型 −18.8'),
    ('C8986',  '中偏负', '**盲测**。无卤池里最负端（−15.4，未标注），验模型"无卤强负"这一块'),
    ('C8313',  '中',     '**盲测**。无卤小分子（MW 290，smi_len 31），中段代表（−11.7）'),
    ('C10640', '正',     '**盲测**。无卤正端代表（+35.2），全库正端最大的一档'),
]
rows = []
for cid, tier, why in SET:
    r = V[V['编号'] == cid]
    if r.empty:
        print(f'  !! {cid} 不在库中，跳过'); continue
    r = r.iloc[0]
    rows.append({'编号': cid, '名称': r['名称'], '档位': tier,
                 '预测_v7': round(float(r['预测_v7']), 2),
                 'CI下_v7': round(float(r['CI下_v7']), 2), 'CI上_v7': round(float(r['CI上_v7']), 2),
                 'CI跨零': bool(r['CI下_v7'] < 0 < r['CI上_v7']),
                 'xTB标签_v7': (round(float(r['xTB标签_v7']), 1) if pd.notna(r['xTB标签_v7']) else None),
                 'xTB标签_v6': (round(float(r['xTB标签_v6']), 1) if pd.notna(r['xTB标签_v6']) else None),
                 '是否盲测': '是' if pd.isna(r['xTB标签_v7']) else '否（已标注，作锚点）',
                 '分子量': round(float(r['分子量']), 1), '无卤': bool(r['无卤']),
                 '选取理由': why, 'SMILES': r['SMILES']})
rs = pd.DataFrame(rows)
rs.to_csv('fix_20260928/exp1_ranking_set_v7.csv', index=False)
print(rs[['编号', '档位', '预测_v7', 'CI下_v7', 'CI上_v7', 'xTB标签_v7', 'xTB标签_v6',
          '是否盲测', '分子量']].to_string(index=False))
print('\n预测跨度:', round(rs['预测_v7'].min(), 2), '→', round(rs['预测_v7'].max(), 2),
      '| 盲测支数:', int((rs['是否盲测'] == '是').sum()), '/', len(rs))
print('saved -> fix_20260928/exp1_ranking_set_v7.csv')

# ---- 附：v6 时代的旧 5 支在 v7 口径下长什么样（用于解释为什么换人）----
old5 = ['C6533', 'C12225', 'C8188', 'C6488', 'C2841']
rows = []
for cid in old5:
    r = V[V['编号'] == cid]
    if r.empty: continue
    r = r.iloc[0]
    rows.append({'编号': cid,
                 'v7预测': round(float(r['预测_v7']), 2),
                 'v7标签': (round(float(r['xTB标签_v7']), 1) if pd.notna(r['xTB标签_v7']) else None),
                 'v6标签': (round(float(r['xTB标签_v6']), 1) if pd.notna(r['xTB标签_v6']) else None),
                 '标签漂移': (round(float(r['xTB标签_v7'] - r['xTB标签_v6']), 1)
                              if pd.notna(r['xTB标签_v7']) and pd.notna(r['xTB标签_v6']) else None),
                 '是否仍盲测': '是' if pd.isna(r['xTB标签_v7']) else '否'})
od = pd.DataFrame(rows)
od.to_csv('fix_20260928/exp1_ranking_set_v6_in_v7.csv', index=False)
print('\n===== v6 旧排序组在 v7 口径下的数值（说明为何换人）=====')
print(od.to_string(index=False))
print('saved -> fix_20260928/exp1_ranking_set_v6_in_v7.csv')
