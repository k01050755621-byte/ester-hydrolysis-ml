# -*- coding: utf-8 -*-
"""AL Round 4 查询：300 个
  ① exploit 100 —— v4 预测最低（最有利水解，前沿加密）
  ② uncertain 100 —— conformal 区间最宽（模型最不确定）
  ③ coverage 100 —— **欠采样化学类优先 + Butina 多样性代表**（补适用域盲区）
排除：已标注 452 + 历史查询（AL3）
输出：ml_final/al4_query.csv
"""
import os
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'
N_EXPLOIT, N_UNC, N_COV = 100, 100, 100

V = pd.read_csv(f'{OUT}/vlib_predictions_v4.csv')          # 编号/名称/SMILES/预测_v4/q90_v4
LAB = pd.read_csv(f'{OUT}/labels_452.csv')
done = set(LAB['编号'])
for f in ['al_round3_query.csv', 'al_round2_query.csv']:
    if os.path.exists(f):
        done |= set(pd.read_csv(f)['编号'])
pool = V[~V['编号'].isin(done)].copy()
pool['CI宽'] = 2 * pool['q90_v4']
print(f'已标注 {len(done)} | 候选池 {len(pool)}')

FV = pd.read_csv(f'{OUT}/feat_vlib.csv')[['编号', 'n_acylAr', 'n_alkAr']]
pool = pool.merge(FV, on='编号', how='left')
pool['cls'] = pool.apply(lambda r: f"{'Ar' if r['n_acylAr'] > 0 else 'Al'}-"
                                  f"{'Ar' if r['n_alkAr'] > 0 else 'Al'}", axis=1)
lab_cls = LAB['编号'].map(dict(zip(FV['编号'], FV.apply(
    lambda r: f"{'Ar' if r['n_acylAr'] > 0 else 'Al'}-{'Ar' if r['n_alkAr'] > 0 else 'Al'}", axis=1)))).value_counts()
print('已标注的类别分布:'); print(lab_cls.to_string())

# ① exploit
exploit = pool.nsmallest(N_EXPLOIT, '预测_v4')
# ② uncertain
rest = pool.drop(exploit.index)
uncertain = rest.nlargest(N_UNC, 'CI宽')
# ③ coverage：按类别配额（欠采样类多取）+ 类内 Butina 多样
rest2 = rest.drop(uncertain.index).copy()
quota = {'Al-Al': 40, 'Al-Ar': 40, 'Ar-Al': 25, 'Ar-Ar': 15}
cov_idx = []
for c, qn in quota.items():
    sub = rest2[rest2['cls'] == c]
    if len(sub) == 0: continue
    sub = sub.sample(frac=1.0, random_state=42)
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=1024) for s in sub['SMILES']]
    n = len(fps); dists = []
    for i in range(1, n):
        dists.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    cl = Butina.ClusterData(dists, n, 0.35, isDistData=True)
    cov_idx += [sub.index[c0[0]] for c0 in cl[:qn]]
coverage = rest2.loc[[i for i in cov_idx if i in rest2.index]]
print(f'采出：exploit {len(exploit)} / uncertain {len(uncertain)} / coverage {len(coverage)}')

q = pd.concat([exploit, uncertain, coverage]).drop_duplicates('编号')
q['query类型'] = (['exploit']*len(exploit) + ['uncertain']*len(uncertain) + ['coverage']*len(coverage))[:len(q)]
q = q[['编号', '名称', 'SMILES', '预测_v4', 'q90_v4', '分子量', 'query类型']]
q.to_csv(f'{OUT}/al4_query.csv', index=False)
print(f'AL4 查询共 {len(q)} 个 → {OUT}/al4_query.csv')
print(q.groupby('query类型')['预测_v4'].describe().round(1).to_string())
