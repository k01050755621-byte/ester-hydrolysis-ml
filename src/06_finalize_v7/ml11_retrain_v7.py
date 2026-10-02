# -*- coding: utf-8 -*-
"""ML 补强 · 第 11 步：v7 标签重训 + v6/v7 同协议 A/B 对比

背景
----
v6 的 1099 个标签由**单构象 GFN2** 生成：嵌入用 ETKDGv3() 而 randomSeed 默认为 -1
（随机），同一分子两次嵌入最大原子位移 7.37 A；经单构象优化后 ΔE 抽样
sigma ~ 11.8 kJ/mol（范围 -30 ~ +15），而标签分布 std 仅 15.4。

v7 协议（fix_20260928/relabel_v7/relabel_v7.py）：
  固定种子 6 构象 -> 各自 GFN2 优化 -> **每物种取最低能量** -> 同热化学式求 ΔE；
  水单独多构象取最低共用。

本脚本把**完全相同的评估协议**分别跑在两组标签上，避免"换标签同时换协议"的混淆：
  A) 旧标签（单构象 GFN2，= v6 口径）
  B) v7 标签（多构象最低）

输入
----
fix_20260928/relabel_v7/labels_v7.csv   编号, v7_kJmol, 旧标签_kJmol, 差_新旧, 来源
fix_20260928/ml_final/feat_vlib.csv     库特征（12,380）
fix_20260928/ml_final/feat_lit.csv      文献分子特征
fix_20260928/ml_final/feature_manifest.json

输出（ml_final/ 下）
----
labels_v7_final.csv           v7 标签与旧标签对照
label_ladder_v7.csv           标签量阶梯（v7 口径）
v6_vs_v7.csv                  两组标签在相同协议下的逐项对比
vlib_predictions_v7.csv       全库 v7 预测 + conformal 区间
pred_v6_vs_v7_library.csv     全库 v6/v7 预测对照
v7_summary.json               汇总
figs/fig10_v6_vs_v7.png       A/B 对比图
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

HOME = os.path.expanduser('~/polywork'); os.chdir(HOME)
OUT = 'fix_20260928/ml_final'
FIG = f'{OUT}/figs'; os.makedirs(FIG, exist_ok=True)

# ---------------- 特征 ----------------
FV = pd.read_csv(f'{OUT}/feat_vlib.csv'); FL = pd.read_csv(f'{OUT}/feat_lit.csv')
MAN = json.load(open(f'{OUT}/feature_manifest.json'))
BASE, MECH = MAN['base_cols'], MAN['mech_cols']
FP = [c for c in FV.columns if c.startswith('fp') and FV[c].sum() > 0]
MK = [c for c in FV.columns if c.startswith('mk') and FV[c].sum() > 0]
BEST = BASE + MECH + FP + MK
ALL = pd.concat([FV, FL], ignore_index=True).fillna(0)

# ---------------- 标签：v6（旧）与 v7 ----------------
LV = pd.read_csv('fix_20260928/relabel_v7/labels_v7.csv')
LV = LV[LV['v7_kJmol'].notna() & LV['旧标签_kJmol'].notna()].copy()
LV = LV.drop_duplicates('编号', keep='last')
LV = LV.rename(columns={'v7_kJmol': 'dE_v7', '旧标签_kJmol': 'dE_old'})
LV['差_新旧'] = LV['dE_v7'] - LV['dE_old']
LV['来源'] = LV['来源'].astype(str)
print(f'v7 标签文件: {len(LV)} 个分子（去重后）')
print(f'新旧差: 均值 {LV["差_新旧"].mean():+.2f}  MAE {LV["差_新旧"].abs().mean():.2f}  '
      f'最大 {LV["差_新旧"].abs().max():.2f}  |  相关系数 {LV[["dE_v7","dE_old"]].corr().iloc[0,1]:.3f}')
print(f'旧标签 std {LV["dE_old"].std():.2f} -> v7 标签 std {LV["dE_v7"].std():.2f}')
LV[['编号', 'dE_v7', 'dE_old', '差_新旧', '来源']].to_csv(f'{OUT}/labels_v7_final.csv', index=False)

# ---------------- 训练表（先合并，再据此算骨架）----------------
TR = ALL.merge(LV[['编号', 'dE_old', 'dE_v7', '来源']], on='编号', how='inner').reset_index(drop=True)
print(f'训练表: {len(TR)} 行 x {len(BEST)} 特征')
X = TR[BEST].values
y_old = TR['dE_old'].values
y_v7 = TR['dE_v7'].values
SRC = TR['来源'].values

# ---------------- 骨架分组 ----------------
vmap = pd.read_csv('vlib_final.csv').set_index('编号')['SMILES'].to_dict()
d1map = pd.read_csv('dataset_v1.csv').set_index('编号')['SMILES'].to_dict()
def scaf_of(s):
    m = Chem.MolFromSmiles(str(s)); sc = MurckoScaffold.GetScaffoldForMol(m) if m is not None else None
    return Chem.MolToSmiles(sc) if sc and sc.GetNumAtoms() else 'acyclic:' + Chem.MolToSmiles(m)
SC = np.array([scaf_of(vmap.get(i, d1map.get(i, 'C'))) for i in TR['编号']])
UQ = np.unique(SC)
def folds_of(seed):
    rng = np.random.RandomState(seed); order = rng.permutation(len(UQ)); fo, sz = {}, np.zeros(5)
    for u in order:
        f = int(np.argmin(sz)); fo[UQ[u]] = f; sz[f] += int(np.sum(SC == UQ[u]))
    return np.array([fo[x] for x in SC])

MODELS = lambda s: [XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                 colsample_bytree=0.7, random_state=s, n_jobs=4),
                    RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=4),
                    HistGradientBoostingRegressor(random_state=s)]
SEEDS_CV = [0, 1, 42]
SEEDS_FIT = (0, 1, 42, 7, 2024)

# ---------------- Butina 严分组（更严于骨架）----------------
def butina_groups():
    from rdkit.Chem import AllChem, DataStructs
    from rdkit.ML.Cluster import Butina
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(vmap.get(i, d1map.get(i, 'C'))),
                                                 2, nBits=1024) for i in TR['编号']]
    n = len(fps); ds = []
    for i in range(1, n):
        ds.extend([1 - x for x in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    cl = Butina.ClusterData(ds, n, 0.45, isDistData=True)
    grp = np.zeros(n, dtype=int)
    for k, c in enumerate(cl):
        for m_ in c:
            grp[m_] = k
    return grp, len(cl), int(max(len(c) for c in cl))

# ---------------- 评估协议（对任意标签向量都同一套）----------------
def cv_eval(y, tag, grp):
    X = TR[BEST].values
    ladder = {}
    stages = [('302', ['302']), ('452', ['302', 'AL3']), ('771', ['302', 'AL3', 'AL4']),
              ('final', sorted(set(SRC.tolist())))]
    for st, srcs in stages:
        mask = np.isin(SRC, srcs)
        mae, r2, rho, bmae = [], [], [], []
        for s in SEEDS_CV:
            f = folds_of(s); Xm, ym, fm = X[mask], y[mask], f[mask]
            p = np.zeros(len(ym)); pb = np.zeros(len(ym))
            for j in range(5):
                te, tri = np.where(fm == j)[0], np.where(fm != j)[0]
                p[te] = np.mean([m.fit(Xm[tri], ym[tri]).predict(Xm[te]) for m in MODELS(s)], axis=0)
                pb[te] = ym[tri].mean()
            mae.append(mean_absolute_error(ym, p)); r2.append(r2_score(ym, p))
            rho.append(spearmanr(p, ym)[0]); bmae.append(mean_absolute_error(ym, pb))
        ladder[st] = {'n': int(mask.sum()), 'MAE': round(float(np.mean(mae)), 2),
                      'R2': round(float(np.mean(r2)), 3), 'rho': round(float(np.mean(rho)), 3),
                      'base_MAE': round(float(np.mean(bmae)), 2)}
        print(f'  [{tag}] {st:6s} n={ladder[st]["n"]:4d}  MAE={ladder[st]["MAE"]:6.2f}  '
              f'R2={ladder[st]["R2"]:+.3f}  rho={ladder[st]["rho"]:.3f}  base={ladder[st]["base_MAE"]:.2f}')
    # y 随机化对照
    yr = y.copy(); np.random.RandomState(0).shuffle(yr)
    mae_r = []
    for s in SEEDS_CV:
        f = folds_of(s); p = np.zeros(len(yr))
        for j in range(5):
            te, tri = np.where(f == j)[0], np.where(f != j)[0]
            p[te] = np.mean([m.fit(X[tri], yr[tri]).predict(X[te]) for m in MODELS(s)], axis=0)
        mae_r.append(mean_absolute_error(yr, p))
    ladder['y_random'] = {'MAE': round(float(np.mean(mae_r)), 2), 'sd': round(float(np.std(mae_r)), 3)}
    print(f'  [{tag}] y-随机化 MAE = {ladder["y_random"]["MAE"]:.2f} +- {ladder["y_random"]["sd"]:.2f}')
    # Butina 严分组
    maes = []
    for s in SEEDS_CV:
        p = np.zeros(len(y))
        for k in range(5):
            te = np.where(grp % 5 == k)[0]; tri = np.where(grp % 5 != k)[0]
            if len(te) == 0: continue
            p[te] = np.mean([m.fit(X[tri], y[tri]).predict(X[te]) for m in MODELS(s)], axis=0)
        maes.append(mean_absolute_error(y, p))
    ladder['butina'] = {'MAE': round(float(np.mean(maes)), 2)}
    print(f'  [{tag}] Butina(0.45) 严分组 MAE = {ladder["butina"]["MAE"]:.2f}')
    return ladder

print('\n计算 Butina 分组 ...')
GRP, NCL, MAXC = butina_groups()
print(f'  Butina(0.45): {NCL} 簇，最大簇 {MAXC}')

print('\n=== A) 旧标签（单构象 GFN2，v6 口径）===')
ladder_old = cv_eval(y_old, 'v6', GRP)
print('\n=== B) v7 标签（多构象最低）===')
ladder_v7 = cv_eval(y_v7, 'v7', GRP)
pd.DataFrame(ladder_v7).T.to_csv(f'{OUT}/label_ladder_v7.csv')

# ---------------- 逐项对比表 ----------------
rows = []
tagmap = {'302': '初始 302', '452': '+AL3', '771': '+AL4', 'final': '+AL5（全部）'}
for st, nm in tagmap.items():
    rows.append({'阶段': nm, 'n': ladder_v7[st]['n'],
                 'v6_MAE': ladder_old[st]['MAE'], 'v7_MAE': ladder_v7[st]['MAE'],
                 'v6_R2': ladder_old[st]['R2'], 'v7_R2': ladder_v7[st]['R2'],
                 'v6_rho': ladder_old[st]['rho'], 'v7_rho': ladder_v7[st]['rho'],
                 'v6基准MAE': ladder_old[st]['base_MAE'], 'v7基准MAE': ladder_v7[st]['base_MAE']})
rows.append({'阶段': 'y-随机化', 'n': len(TR),
             'v6_MAE': ladder_old['y_random']['MAE'], 'v7_MAE': ladder_v7['y_random']['MAE'],
             'v6_R2': np.nan, 'v7_R2': np.nan, 'v6_rho': np.nan, 'v7_rho': np.nan,
             'v6基准MAE': np.nan, 'v7基准MAE': np.nan})
rows.append({'阶段': f'Butina 严分组（{NCL} 簇）', 'n': len(TR),
             'v6_MAE': ladder_old['butina']['MAE'], 'v7_MAE': ladder_v7['butina']['MAE'],
             'v6_R2': np.nan, 'v7_R2': np.nan, 'v6_rho': np.nan, 'v7_rho': np.nan,
             'v6基准MAE': np.nan, 'v7基准MAE': np.nan})
CMP = pd.DataFrame(rows)
CMP.to_csv(f'{OUT}/v6_vs_v7.csv', index=False)
print('\n=== v6 vs v7 同协议对比 ===')
print(CMP.to_string(index=False))

# ---------------- v7 最终模型 -> 全库 ----------------
print('\n=== v7 最终模型 -> 全库 ===')
X = TR[BEST].values
def fit_pred(Xa, ya, Xb, seeds=SEEDS_FIT):
    return np.mean([m.fit(Xa, ya).predict(Xb) for s in seeds for m in MODELS(s)], axis=0)
pred = fit_pred(X, y_v7, FV[BEST].values)
V = pd.read_csv('vlib_final.csv')[['编号', '名称', 'SMILES', '分子量']].copy()
V['预测_v7'] = pred

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
print(f'  q90 = {q90:.2f} | 覆盖率（名义 90%）= {cov:.1%} | 各划分 {QS}')
V['q90_v7'] = q90; V['CI下_v7'] = V['预测_v7'] - q90; V['CI上_v7'] = V['预测_v7'] + q90
def nhal(s):
    m = Chem.MolFromSmiles(s); return sum(1 for a in m.GetAtoms() if a.GetSymbol() in ('F', 'Cl', 'Br', 'I'))
V['n_hal'] = [nhal(s) for s in V['SMILES']]
V[['编号', '名称', 'SMILES', '分子量', '预测_v7', 'q90_v7', 'CI下_v7', 'CI上_v7', 'n_hal']].to_csv(
    f'{OUT}/vlib_predictions_v7.csv', index=False)
print(f'  不同预测值 {V["预测_v7"].round(4).nunique()}/{len(V)} | 区间全负 {int((V["CI上_v7"] < 0).sum())}')

rho_lib = float('nan')
try:
    v6 = pd.read_csv(f'{OUT}/vlib_predictions_v6.csv')[['编号', '预测_v6']]
    MRG = V.merge(v6, on='编号', how='left')
    MRG['预测差_v7_v6'] = MRG['预测_v7'] - MRG['预测_v6']
    MRG[['编号', '名称', '预测_v6', '预测_v7', '预测差_v7_v6', '分子量', 'n_hal']].sort_values(
        '预测_v7').to_csv(f'{OUT}/pred_v6_vs_v7_library.csv', index=False)
    rho_lib = float(spearmanr(MRG['预测_v6'], MRG['预测_v7'])[0])
    print(f'  全库 v6 vs v7 预测 Spearman = {rho_lib:.3f}')
except Exception as e:
    print(f'  （v6 预测对照跳过：{e}）')

# ---------------- 出图 ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].scatter(LV['dE_old'], LV['dE_v7'], s=6, alpha=.45, color='#4C72B0', edgecolors='none')
lo = min(LV['dE_old'].min(), LV['dE_v7'].min()) - 5
hi = max(LV['dE_old'].max(), LV['dE_v7'].max()) + 5
ax[0].plot([lo, hi], [lo, hi], 'k--', lw=.8)
ax[0].axhline(0, color='gray', lw=.5); ax[0].axvline(0, color='gray', lw=.5)
ax[0].set_xlabel(r'Single-conformer GFN2 $\Delta E$ (kJ mol$^{-1}$)')
ax[0].set_ylabel(r'Multi-conformer (v7) $\Delta E$ (kJ mol$^{-1}$)')
ax[0].set_title(f'(a) Label relabelling (n={len(LV)})')
ax[0].text(.04, .96, f'MAE diff {LV["差_新旧"].abs().mean():.1f}\nmax {LV["差_新旧"].abs().max():.1f} kJ/mol',
           transform=ax[0].transAxes, va='top', fontsize=7)
xs = np.arange(len(tagmap)); w = .38
ax[1].bar(xs - w/2, [ladder_old[s]['R2'] for s in tagmap], w, color='#C44E52', label='v6 labels (single conf.)')
ax[1].bar(xs + w/2, [ladder_v7[s]['R2'] for s in tagmap], w, color='#4C72B0', label='v7 labels (multi-conf.)')
for i, s in enumerate(tagmap):
    ax[1].text(i - w/2, ladder_old[s]['R2'] + .015, f'{ladder_old[s]["R2"]:.2f}', ha='center', fontsize=6.5)
    ax[1].text(i + w/2, ladder_v7[s]['R2'] + .015, f'{ladder_v7[s]["R2"]:.2f}', ha='center', fontsize=6.5)
EN = {'302': '302', '452': '+AL3', '771': '+AL4', 'final': 'full'}
ax[1].set_xticks(xs); ax[1].set_xticklabels([EN[s] for s in tagmap], fontsize=7)
ax[1].set_ylabel(r'R$^2$ (scaffold-grouped CV)'); ax[1].set_title('(b) Same protocol, two label sets')
ax[1].legend(frameon=False, fontsize=7); ax[1].axhline(0, color='gray', lw=.8)
fig.tight_layout(); fig.savefig(f'{FIG}/fig10_v6_vs_v7.png', bbox_inches='tight'); plt.close(fig)

json.dump({'n_labels': int(len(LV)),
           'label_diff': {'mean': round(float(LV['差_新旧'].mean()), 2),
                          'mae': round(float(LV['差_新旧'].abs().mean()), 2),
                          'max_abs': round(float(LV['差_新旧'].abs().max()), 2),
                          'pearson': round(float(LV[['dE_v7', 'dE_old']].corr().iloc[0, 1]), 3)},
           'old_std': round(float(LV['dE_old'].std()), 2), 'v7_std': round(float(LV['dE_v7'].std()), 2),
           'ladder_v6': ladder_old, 'ladder_v7': ladder_v7,
           'butina': {'n_clusters': NCL, 'max_cluster': MAXC},
           'q90': round(q90, 2), 'coverage_test': round(cov, 3), 'conformal_splits': QS,
           'lib_v6_v7_spearman': round(rho_lib, 3),
           'n_distinct_pred': int(V['预测_v7'].round(4).nunique()),
           'n_CI_all_negative': int((V['CI上_v7'] < 0).sum())},
          open(f'{OUT}/v7_summary.json', 'w'), ensure_ascii=False, indent=2, default=str)
print('\n完成 -> v7_summary.json / v6_vs_v7.csv / vlib_predictions_v7.csv / figs/fig10_v6_vs_v7.png')
