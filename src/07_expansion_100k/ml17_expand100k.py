# -*- coding: utf-8 -*-
"""扩库 10⁵ 的特征构建 + v7 模型预测。

流程：
  1) 复刻 ml11_retrain_v7.py 的最终模型训练（同一标签文件、同一 BEST 特征、
     同一集成与种子）——先对原 12,380 库预测并与 vlib_predictions_v7.csv
     逐值核对（必须完全一致，否则中止）；
  2) 复刻 ml11 的 split3 conformal，复核 q90（应 = 14.53）；
  3) 用 build_features.feats 给 v4 的 87,620 个新分子建特征（落盘 feat_vlib4.csv）；
  4) 预测 + q90 区间 + n_hal，合并成 10 万全库预测表 vlib_100k_predictions_v7.csv；
  5) 统计：分布、CI 全负数、无卤池、新 Top、同分异构体区分度、与旧库的
     重叠检查；写 expand100k_summary.json。
"""
import os, sys, json, time
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'
sys.path.insert(0, OUT)
from build_features import feats   # 特征工厂（与训练时完全同一份代码）

# ---------------- 特征与标签（复刻 ml11） ----------------
FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)

LV = pd.read_csv('fix_20260928/relabel_v7/labels_v7.csv')
LV = LV[LV['v7_kJmol'].notna() & LV['旧标签_kJmol'].notna()].copy()
LV = LV.drop_duplicates('编号', keep='last')
LV = LV.rename(columns={'v7_kJmol': 'dE_v7', '旧标签_kJmol': 'dE_old'})
LV['来源'] = LV['来源'].astype(str)
TR = ALL.merge(LV[['编号', 'dE_v7', '来源']], on='编号', how='inner').reset_index(drop=True)
X = TR[BEST].values; y_v7 = TR['dE_v7'].values
print(f'训练表: {len(TR)} 行 x {len(BEST)} 特征', flush=True)

vmap = pd.read_csv('vlib_final.csv').set_index('编号')['SMILES'].to_dict()
d1map = pd.read_csv('dataset_v1.csv').set_index('编号')['SMILES'].to_dict()
def scaf_of(s):
    m = Chem.MolFromSmiles(str(s)); sc = MurckoScaffold.GetScaffoldForMol(m) if m is not None else None
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)
SC = np.array([scaf_of(vmap.get(i, d1map.get(i, 'C'))) for i in TR['编号']])
UQ = np.unique(SC)

MODELS = lambda s: [XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.7, random_state=s, n_jobs=4),
                    RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4),
                    HistGradientBoostingRegressor(random_state=s)]
SEEDS_FIT = (0, 1, 42, 7, 2024)
def fit_pred(Xa, ya, Xb, seeds=SEEDS_FIT):
    return np.mean([m.fit(Xa, ya).predict(Xb) for s in seeds for m in MODELS(s)], axis=0)

# ---------------- 1) 复刻核对：原库预测必须与 vlib_predictions_v7 完全一致 ----------------
print('复刻最终模型并对原 12,380 库预测核对...', flush=True)
pred_old = fit_pred(X, y_v7, FV[BEST].values)
REF = pd.read_csv(f'{OUT}/vlib_predictions_v7.csv')
ref_map = dict(zip(REF['编号'], REF['预测_v7']))
ref_arr = np.array([ref_map[i] for i in FV['编号']])
dmax = float(np.max(np.abs(pred_old - ref_arr)))
print(f'  与 vlib_predictions_v7 的最大偏差 = {dmax:.2e}')
assert dmax < 1e-6, '模型复刻失败：与已发布预测不一致，中止！'

# ---------------- 2) q90 复核 ----------------
def split3(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UQ))
    tgt = np.array([.5, .25, .25]) * len(TR); got = np.zeros(3); fo = {}
    for u in order:
        f = int(np.argmin(got / tgt)); fo[UQ[u]] = f; got[f] += int(np.sum(SC == UQ[u]))
    lb = np.array([fo[x] for x in SC]); return np.where(lb == 0)[0], np.where(lb == 1)[0], np.where(lb == 2)[0]
QS = []
for sd in [42, 7, 2024]:
    tri, cal, tst = split3(sd)
    rc = np.abs(y_v7[cal] - fit_pred(X[tri], y_v7[tri], X[cal], seeds=(42,)))
    q = float(np.quantile(rc, .9)); rt = np.abs(y_v7[tst] - fit_pred(X[tri], y_v7[tri], X[tst], seeds=(42,)))
    QS.append({'q90': round(q, 2), 'coverage': round(float((rt <= q).mean()), 3)})
q90 = float(np.mean([x['q90'] for x in QS])); cov = float(np.mean([x['coverage'] for x in QS]))
print(f'  q90 = {q90:.2f} | 覆盖率 = {cov:.1%} | 各划分 {QS}', flush=True)

# ---------------- 3) v4 新分子特征 ----------------
V4 = pd.read_csv('枚举库_ester_v4.csv')
assert len(V4) == 87620, f'v4 行数异常 {len(V4)}'
assert not set(V4['编号']) & set(FV['编号']), '与原库编号冲突！'
t0 = time.time(); rows = []
for k, (i, s) in enumerate(zip(V4['编号'], V4['SMILES']), 1):
    f = feats(s); f['编号'] = i; rows.append(f)
    if k % 10000 == 0:
        print(f'  特征 {k}/87620 | {k/(time.time()-t0):.0f} mol/s', flush=True)
F4 = pd.DataFrame(rows).fillna(0)
assert len(F4) == len(V4), f'特征行数 {len(F4)} != 分子数 {len(V4)}'
F4.to_csv(f'{OUT}/feat_vlib4.csv', index=False)
print(f'feat_vlib4.csv 落盘: {F4.shape}', flush=True)

# ---------------- 4) 预测 + 合并 ----------------
pred4 = fit_pred(X, y_v7, F4[BEST].values)
def nhal(s):
    m = Chem.MolFromSmiles(s); return sum(1 for a in m.GetAtoms() if a.GetSymbol() in ('F','Cl','Br','I'))
NEW = pd.DataFrame({'编号': V4['编号'], '名称': V4['名称'], 'SMILES': V4['SMILES']})
NEW['分子量'] = F4['分子量'].values
NEW['预测_v7'] = pred4; NEW['q90_v7'] = q90
NEW['CI下_v7'] = NEW['预测_v7'] - q90; NEW['CI上_v7'] = NEW['预测_v7'] + q90
NEW['n_hal'] = [nhal(s) for s in NEW['SMILES']]
OLD = REF.copy(); OLD['库'] = 'v1-3'
NEW['库'] = 'v4'
FULL = pd.concat([OLD[['编号','名称','SMILES','分子量','预测_v7','q90_v7','CI下_v7','CI上_v7','n_hal','库']],
                  NEW[['编号','名称','SMILES','分子量','预测_v7','q90_v7','CI下_v7','CI上_v7','n_hal','库']]],
                 ignore_index=True)
assert len(FULL) == 100000 and FULL['SMILES'].nunique() == 100000, '合并库规模/唯一性异常！'
FULL.to_csv(f'{OUT}/vlib_100k_predictions_v7.csv', index=False)

# ---------------- 5) 统计 ----------------
p_new = NEW['预测_v7'].values; p_old = OLD['预测_v7'].values
stats = {
    'n_total': 100000, 'n_old': 12380, 'n_new': 87620,
    'q90': round(q90, 2), 'coverage': round(cov, 3),
    'pred_new': {'mean': round(float(p_new.mean()), 2), 'sd': round(float(p_new.std()), 2),
                 'min': round(float(p_new.min()), 1), 'max': round(float(p_new.max()), 1)},
    'pred_old': {'mean': round(float(p_old.mean()), 2), 'sd': round(float(p_old.std()), 2),
                 'min': round(float(p_old.min()), 1), 'max': round(float(p_old.max()), 1)},
    'n_distinct_pred_100k': int(FULL['预测_v7'].round(4).nunique()),
    'n_CI_all_negative_100k': int((FULL['CI上_v7'] < 0).sum()),
    'n_CI_all_negative_new': int((NEW['CI上_v7'] < 0).sum()),
    'n_halogen_free_100k': int((FULL['n_hal'] == 0).sum()),
    'v4_attempt': None,
}
TOP = NEW.sort_values('预测_v7').head(15)
stats['new_top15'] = TOP[['编号','名称','预测_v7','CI上_v7','分子量','n_hal']].to_dict('records')
# 与旧库的区分度抽查：新库里预测最负 10 个的名称结构
json.dump(stats, open(f'{OUT}/expand100k_summary.json', 'w'), ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in stats.items() if k != 'new_top15'},
                 ensure_ascii=False, indent=2), flush=True)
print('\n新库最负 Top10：')
print(TOP.head(10)[['编号','名称','预测_v7','CI上_v7','分子量','n_hal']].to_string(index=False))
print('\n完成 -> feat_vlib4.csv / vlib_100k_predictions_v7.csv / expand100k_summary.json')
