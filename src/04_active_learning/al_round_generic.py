# -*- coding: utf-8 -*-
"""通用 AL 轮次查询：python al_round_generic.py <round> <pred_csv> <pred_col> <n_each>
例：python al_round_generic.py 5 vlib_predictions_v5.csv 预测_v5 110

三层：exploit（预测最低）/ uncertain（区间最宽）/ coverage（欠采样类 + Butina 多样）
排除：所有已标注 + 历史查询文件。
"""
import os, sys, glob
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')

RN = int(sys.argv[1]); PRED_CSV = sys.argv[2]; PRED_COL = sys.argv[3]; N = int(sys.argv[4])
HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'

V = pd.read_csv(f'{OUT}/{PRED_CSV}')
LAB = pd.read_csv(f'{OUT}/labels_771.csv' if os.path.exists(f'{OUT}/labels_771.csv') else f'{OUT}/labels_772.csv')
done = set(LAB['编号'])
for f in glob.glob(f'{OUT}/al*_query.csv') + glob.glob('al_round*_query.csv'):
    done |= set(pd.read_csv(f)['编号'])
pool = V[~V['编号'].isin(done)].copy()
pool['CI宽'] = pool['CI上_v5'] - pool['CI下_v5'] if 'CI上_v5' in pool else 2 * pool['q90_v4']
print(f'已标注+已查询 {len(done)} | 候选池 {len(pool)}')

FV = pd.read_csv(f'{OUT}/feat_vlib.csv')[['编号', 'n_acylAr', 'n_alkAr']]
pool = pool.merge(FV, on='编号')
pool['cls'] = pool.apply(lambda r: f"{'Ar' if r['n_acylAr']>0 else 'Al'}-{'Ar' if r['n_alkAr']>0 else 'Al'}", axis=1)

exploit = pool.nsmallest(N, PRED_COL)
rest = pool.drop(exploit.index)
uncertain = rest.nlargest(N, 'CI宽')
rest2 = rest.drop(uncertain.index).copy()
# coverage：按类配额（欠采样类优先）
lab_cls = LAB['编号'].map(dict(zip(FV['编号'], FV.apply(
    lambda r: f"{'Ar' if r['n_acylAr']>0 else 'Al'}-{'Ar' if r['n_alkAr']>0 else 'Al'}", axis=1)))).value_counts()
tot = lab_cls.sum()
quota = {}
for c in ['Al-Al', 'Al-Ar', 'Ar-Al', 'Ar-Ar']:
    need = max(0, int(tot / 4) - int(lab_cls.get(c, 0)))
    quota[c] = min(N, max(15, need))
# 归一化到 N
s = sum(quota.values())
if s > 0:
    quota = {k: max(8, int(round(v * N / s))) for k, v in quota.items()}
print('coverage 配额:', quota)
cov_idx = []
for c, qn in quota.items():
    sub = rest2[rest2['cls'] == c]
    if len(sub) == 0: continue
    sub = sub.sample(frac=1.0, random_state=RN)
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in sub['SMILES']]
    n = len(fps); dists = []
    for i in range(1, n):
        dists.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    cl = Butina.ClusterData(dists, n, 0.35, isDistData=True)
    cov_idx += [sub.index[c0[0]] for c0 in cl[:qn]]
coverage = rest2.loc[[i for i in cov_idx if i in rest2.index]]

q = pd.concat([exploit, uncertain, coverage]).drop_duplicates('编号')
q['query类型'] = (['exploit']*len(exploit) + ['uncertain']*len(uncertain) + ['coverage']*len(coverage))[:len(q)]
q90col = 'q90_v5' if 'q90_v5' in q.columns else 'q90_v4'
q = q[['编号', '名称', 'SMILES', PRED_COL, q90col, '分子量', 'query类型']]
q.columns = ['编号', '名称', 'SMILES', '模型预测', 'q90', '分子量', 'query类型']
q.to_csv(f'{OUT}/al{RN}_query.csv', index=False)
print(f'AL{RN} 查询 {len(q)} 个 → {OUT}/al{RN}_query.csv')
print(q.groupby('query类型')['模型预测'].describe().round(1).to_string())
